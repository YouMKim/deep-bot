# Phase 4: Chunking Strategies

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Chunking Strategies

### Learning Objectives
- Understand different chunking approaches
- Learn when to use each strategy
- Design extensible chunking system
- Practice algorithm design
- Implement token-aware chunking for production RAG

### Design Principles
- **Strategy Pattern**: Different chunking algorithms
- **Configurable Parameters**: Window sizes, gaps as config
- **Extensibility**: Easy to add new strategies
- **Production-Ready**: Token limits, validation, error handling

---

## Implementation Steps

### Step 4.0: Add Configuration and Dependencies

**Add chunking configuration to `config.py`:**

```python
# Chunking Configuration
CHUNKING_TEMPORAL_WINDOW: int = int(os.getenv("CHUNKING_TEMPORAL_WINDOW", "3600"))  # 1 hour in seconds
CHUNKING_CONVERSATION_GAP: int = int(os.getenv("CHUNKING_CONVERSATION_GAP", "1800"))  # 30 minutes
CHUNKING_WINDOW_SIZE: int = int(os.getenv("CHUNKING_WINDOW_SIZE", "10"))  # messages per window
CHUNKING_OVERLAP: int = int(os.getenv("CHUNKING_OVERLAP", "2"))  # overlapping messages
CHUNKING_MAX_TOKENS: int = int(os.getenv("CHUNKING_MAX_TOKENS", "512"))  # max tokens per chunk
CHUNKING_MIN_CHUNK_SIZE: int = int(os.getenv("CHUNKING_MIN_CHUNK_SIZE", "3"))  # min messages per chunk
```

**Add tiktoken to `requirements.txt`:**

```txt
tiktoken>=0.5.0
```

**Install the dependency:**

```bash
pip install tiktoken
```

**Update `.env` (optional - defaults are good for most cases):**

```bash
# Chunking Settings
CHUNKING_TEMPORAL_WINDOW=3600
CHUNKING_CONVERSATION_GAP=1800
CHUNKING_WINDOW_SIZE=10
CHUNKING_OVERLAP=2
CHUNKING_MAX_TOKENS=512
CHUNKING_MIN_CHUNK_SIZE=3
```

**⚠️ Important Note on Message Field Names:**

The database schema uses:
- `message_id` (not `id`)
- `author_name` and `author_display_name` (not just `author`)
- `timestamp` (matches)
- `channel_id` (matches)

All code must use these field names when accessing message dictionaries.

---

### Step 4.1: Design Chunk Data Structure

Create `chunking/base.py`:

```python
"""
Chunk data structure for RAG systems.

Learning: A chunk represents a group of related messages that will be
embedded together. Good chunk design is critical for RAG quality.
"""

from typing import List, Dict


class Chunk:
    """
    Data structure for a chunk.

    Learning: Good data modeling is crucial for RAG systems.
    Metadata enables filtering and context preservation.

    A chunk contains:
    - content: The formatted text to embed
    - message_ids: References to source messages
    - metadata: Strategy, timestamps, counts, etc.
    """

    def __init__(
        self,
        content: str,
        message_ids: List[str],
        metadata: Dict
    ):
        self.content = content
        self.message_ids = message_ids
        self.metadata = metadata

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for storage."""
        return {
            "content": self.content,
            "message_ids": self.message_ids,
            "metadata": self.metadata
        }

    def __repr__(self):
        return (
            f"Chunk(messages={len(self.message_ids)}, "
            f"tokens={self.metadata.get('token_count', 'unknown')}, "
            f"strategy={self.metadata.get('chunk_strategy', 'unknown')})"
        )
```

---

### Step 4.2: Create ChunkingService Foundation

Create `chunking/service.py`:

```python
"""
Chunking service with multiple strategies.

Learning: The Strategy pattern allows experimenting with different
chunking approaches without changing the interface.
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging
import tiktoken
from config import Config
from chunking.base import Chunk


class ChunkingService:
    """
    Service for chunking messages using different strategies.

    Learning: Strategy pattern allows experimenting with different approaches.

    Available strategies:
    - temporal: Time-based windows
    - conversation: Gap detection
    - single: One message per chunk
    - sliding_window: Overlapping windows (NEW)
    - token_aware: Respect token limits (NEW)
    """

    def __init__(
        self,
        temporal_window: int = None,
        conversation_gap: int = None
    ):
        self.logger = logging.getLogger(__name__)
        self.temporal_window = temporal_window or Config.CHUNKING_TEMPORAL_WINDOW
        self.conversation_gap = conversation_gap or Config.CHUNKING_CONVERSATION_GAP

        # Initialize tokenizer (lazy loaded)
        self._tokenizer = None

    def _validate_messages(self, messages: List[Dict]) -> bool:
        """
        Validate that messages list is valid for chunking.

        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(messages, list):
            raise ValueError(f"Messages must be a list, got {type(messages)}")

        if not messages:
            return True  # Empty list is valid (will return empty chunks)

        # Check that first message has required fields
        sample = messages[0]
        if not isinstance(sample, dict):
            raise ValueError(f"Messages must be dictionaries, got {type(sample)}")

        return True

    def count_tokens(self, text: str, model: str = "cl100k_base") -> int:
        """
        Count tokens in text using tiktoken.

        Learning: Different models have different tokenizers.
        - cl100k_base: GPT-4, GPT-3.5-turbo
        - p50k_base: GPT-3 (davinci, curie, etc.)

        Why this matters:
        - Embedding models have max token limits (384-8192 tokens)
        - LLMs have context limits (4k-128k tokens)
        - Discord messages can be 2000 chars = ~500 tokens

        Args:
            text: Text to count tokens for
            model: Tokenizer model name

        Returns:
            Number of tokens
        """
        try:
            if self._tokenizer is None:
                self._tokenizer = tiktoken.get_encoding(model)
            return len(self._tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}. Using fallback estimate.")
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4
```

---

### Step 4.3: Implement Temporal Chunking

Add to `chunking/service.py`:

```python
    def chunk_temporal(self, messages: List[Dict]) -> List[Chunk]:
        """
        Group messages by time windows.

        Learning: Temporal chunking preserves time-based context.
        Useful for conversations that happen over time.

        Example:
            Window = 1 hour
            - Chunk 1: Messages from 10:00-11:00
            - Chunk 2: Messages from 11:00-12:00

        Args:
            messages: List of message dictionaries

        Returns:
            List of temporal chunks
        """
        self._validate_messages(messages)
        if not messages:
            return []

        # Sort messages by timestamp
        sorted_messages = sorted(
            messages,
            key=lambda m: m.get('timestamp', '')
        )

        chunks = []
        current_chunk = []
        window_start = None

        for message in sorted_messages:
            try:
                # Parse timestamp
                timestamp_str = message.get('timestamp', '')
                if 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                # Skip messages with invalid timestamps
                self.logger.warning(f"Skipping message with invalid timestamp: {e}")
                continue

            # Start new window if needed
            if window_start is None:
                window_start = timestamp

            # Check if message is outside current window
            time_diff = (timestamp - window_start).total_seconds()

            if time_diff > self.temporal_window:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, "temporal"))
                current_chunk = [message]
                window_start = timestamp
            else:
                current_chunk.append(message)

        # Don't forget last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, "temporal"))

        self.logger.info(f"Temporal chunking created {len(chunks)} chunks from {len(messages)} messages")
        return chunks
```

---

### Step 4.4: Implement Conversation Chunking

Add to `chunking/service.py`:

```python
    def chunk_conversation(self, messages: List[Dict]) -> List[Chunk]:
        """
        Group messages by conversation boundaries.

        Learning: Conversation chunking detects natural breaks in dialogue.
        Boundaries: time gaps, channel changes, topic shifts.

        Example:
            Gap = 30 minutes
            - 10:00 AM: "Hello"
            - 10:01 AM: "Hi there"
            - 10:35 AM: "Different topic" <- NEW CHUNK (gap > 30min)

        Args:
            messages: List of message dictionaries

        Returns:
            List of conversation chunks
        """
        self._validate_messages(messages)
        if not messages:
            return []

        sorted_messages = sorted(
            messages,
            key=lambda m: m.get('timestamp', '')
        )

        chunks = []
        current_chunk = []
        last_timestamp = None

        for message in sorted_messages:
            try:
                timestamp_str = message.get('timestamp', '')
                if 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception:
                continue

            channel_id = message.get('channel_id')

            # Check for conversation boundary
            is_boundary = False

            if last_timestamp:
                time_gap = (timestamp - last_timestamp).total_seconds()
                if time_gap > self.conversation_gap:
                    is_boundary = True

            # Channel change is also a boundary
            if current_chunk and current_chunk[-1].get('channel_id') != channel_id:
                is_boundary = True

            if is_boundary and current_chunk:
                chunks.append(self._create_chunk(current_chunk, "conversation"))
                current_chunk = [message]
            else:
                current_chunk.append(message)

            last_timestamp = timestamp

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, "conversation"))

        self.logger.info(f"Conversation chunking created {len(chunks)} chunks from {len(messages)} messages")
        return chunks
```

---

### Step 4.5: Implement Single Message Chunking

Add to `chunking/service.py`:

```python
    def chunk_single(self, messages: List[Dict]) -> List[Chunk]:
        """
        Each message is its own chunk.

        Learning: Baseline strategy for comparison.
        Maximum granularity, preserves all individual messages.

        Use when:
        - Messages are already well-scoped
        - Maximum search precision needed
        - Comparing against other strategies

        Args:
            messages: List of message dictionaries

        Returns:
            List of single-message chunks
        """
        self._validate_messages(messages)
        chunks = []

        for message in messages:
            try:
                chunks.append(self._create_chunk([message], "single"))
            except Exception as e:
                self.logger.warning(f"Skipping message in single chunking: {e}")
                continue

        self.logger.info(f"Single chunking created {len(chunks)} chunks from {len(messages)} messages")
        return chunks
```

---

### Step 4.6: Implement Sliding Window Chunking

Add to `chunking/service.py`:

```python
    def chunk_sliding_window(
        self,
        messages: List[Dict],
        window_size: int = None,
        overlap: int = None
    ) -> List[Chunk]:
        """
        Create overlapping chunks with sliding window.

        Learning: Overlap prevents losing context at chunk boundaries.
        Critical for RAG to find information that spans multiple messages.

        Example (window=3, overlap=1):
            Chunk 1: [msg1, msg2, msg3]
            Chunk 2: [msg3, msg4, msg5]  # msg3 overlaps
            Chunk 3: [msg5, msg6, msg7]  # msg5 overlaps

        Why overlap matters:
        - Conversations often span chunk boundaries
        - Overlap ensures queries can match across boundaries
        - Slight redundancy improves retrieval quality

        Args:
            messages: List of messages to chunk
            window_size: Number of messages per chunk (default from config)
            overlap: Number of messages to overlap between chunks (default from config)

        Returns:
            List of chunks with overlapping messages
        """
        self._validate_messages(messages)
        if not messages:
            return []

        window_size = window_size or Config.CHUNKING_WINDOW_SIZE
        overlap = overlap or Config.CHUNKING_OVERLAP

        if overlap >= window_size:
            self.logger.warning(
                f"Overlap ({overlap}) >= window_size ({window_size}), "
                f"setting overlap to {window_size - 1}"
            )
            overlap = max(1, window_size - 1)

        sorted_messages = sorted(messages, key=lambda m: m.get('timestamp', ''))
        chunks = []

        start = 0
        while start < len(sorted_messages):
            end = min(start + window_size, len(sorted_messages))
            window_messages = sorted_messages[start:end]

            if window_messages:
                chunks.append(self._create_chunk(window_messages, "sliding_window"))

            # Move forward by (window_size - overlap)
            start += window_size - overlap

            # Prevent infinite loop if we're at the end
            if end == len(sorted_messages):
                break

        self.logger.info(
            f"Sliding window chunking created {len(chunks)} chunks "
            f"from {len(messages)} messages (window={window_size}, overlap={overlap})"
        )
        return chunks
```

---

### Step 4.7: Implement Token-Aware Chunking

Add to `chunking/service.py`:

```python
    def chunk_by_tokens(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        min_chunk_size: int = None
    ) -> List[Chunk]:
        """
        Create chunks respecting token limits.

        Learning: Embedding models have max input sizes:
        - all-MiniLM-L6-v2: 512 tokens
        - all-mpnet-base-v2: 512 tokens
        - OpenAI ada-002: 8192 tokens
        - text-embedding-3-small: 8192 tokens

        Why this matters:
        - Exceeding token limits causes errors or truncation
        - Different strategies produce different chunk sizes
        - Need to validate chunks before embedding

        Args:
            messages: List of messages to chunk
            max_tokens: Maximum tokens per chunk (default from config)
            min_chunk_size: Minimum messages per chunk (default from config)

        Returns:
            List of chunks within token limits
        """
        self._validate_messages(messages)
        if not messages:
            return []

        max_tokens = max_tokens or Config.CHUNKING_MAX_TOKENS
        min_chunk_size = min_chunk_size or Config.CHUNKING_MIN_CHUNK_SIZE

        sorted_messages = sorted(messages, key=lambda m: m.get('timestamp', ''))
        chunks = []
        current_chunk = []
        current_tokens = 0

        for message in sorted_messages:
            # Format message to see actual token count
            author = message.get('author_display_name') or message.get('author_name') or 'Unknown'
            timestamp = message.get('timestamp', '')[:10]
            content = message.get('content', '').strip()

            if not content:
                continue

            formatted = f"{timestamp} - {author}: {content}\n"
            msg_tokens = self.count_tokens(formatted)

            # If single message exceeds limit, skip it
            if msg_tokens > max_tokens:
                # Save current chunk if it exists
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, "token_aware"))
                    current_chunk = []
                    current_tokens = 0

                # Handle oversized message (skip with warning)
                self.logger.warning(
                    f"Message {message.get('message_id', 'unknown')} exceeds {max_tokens} tokens "
                    f"({msg_tokens} tokens). Skipping."
                )
                continue

            # Check if adding this message would exceed limit
            if current_tokens + msg_tokens > max_tokens:
                # Save current chunk if it meets minimum size
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, "token_aware"))
                elif current_chunk:
                    self.logger.warning(
                        f"Chunk too small ({len(current_chunk)} messages), "
                        f"but at token limit. Creating anyway."
                    )
                    chunks.append(self._create_chunk(current_chunk, "token_aware"))

                # Start new chunk
                current_chunk = [message]
                current_tokens = msg_tokens
            else:
                # Add message to current chunk
                current_chunk.append(message)
                current_tokens += msg_tokens

        # Don't forget last chunk
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, "token_aware"))
        elif current_chunk:
            self.logger.warning(
                f"Last chunk too small ({len(current_chunk)} messages). "
                f"Creating anyway to avoid data loss."
            )
            chunks.append(self._create_chunk(current_chunk, "token_aware"))

        self.logger.info(
            f"Token-aware chunking created {len(chunks)} chunks "
            f"from {len(messages)} messages (max_tokens={max_tokens})"
        )
        return chunks
```

---

### Step 4.8: Implement Chunk Creation Helper

Add to `chunking/service.py`:

```python
    def _create_chunk(self, messages: List[Dict], strategy: str) -> Chunk:
        """
        Helper to create chunk with metadata and validation.

        Args:
            messages: List of messages to include in chunk
            strategy: Name of chunking strategy used

        Returns:
            Chunk object with content, IDs, and metadata
        """
        if not messages:
            raise ValueError("Cannot create chunk from empty messages")

        # Format content
        content_parts = []
        for msg in messages:
            # Use author_display_name if available, fallback to author_name, then 'Unknown'
            author = msg.get('author_display_name') or msg.get('author_name') or 'Unknown'
            timestamp = msg.get('timestamp', '')[:10]  # Date only
            content = msg.get('content', '').strip()
            if content:
                content_parts.append(f"{timestamp} - {author}: {content}")

        content = "\n".join(content_parts)

        # Validate token count
        token_count = self.count_tokens(content)

        # Collect metadata - use message_id (database field name)
        message_ids = [str(msg.get('message_id', '')) for msg in messages if msg.get('message_id')]
        metadata = {
            "chunk_strategy": strategy,
            "channel_id": messages[0].get('channel_id', ''),
            "message_count": len(messages),
            "token_count": token_count,
            "first_message_id": message_ids[0] if message_ids else '',
            "last_message_id": message_ids[-1] if message_ids else '',
            "first_timestamp": messages[0].get('timestamp', ''),
            "last_timestamp": messages[-1].get('timestamp', ''),
        }

        # Log warnings for problematic chunks
        if token_count > 512:
            self.logger.warning(
                f"Chunk exceeds 512 tokens ({token_count}). "
                f"May fail with some embedding models."
            )

        if len(messages) == 1:
            self.logger.debug(f"Single-message chunk created (strategy: {strategy})")

        return Chunk(content, message_ids, metadata)
```

---

### Step 4.9: Implement Main Chunking Method

Add to `chunking/service.py`:

```python
    def chunk_messages(
        self,
        messages: List[Dict],
        strategies: List[str] = None
    ) -> Dict[str, List[Chunk]]:
        """
        Generate chunks using specified strategies.

        Learning: Returns all strategies at once for comparison.

        Available strategies:
        - temporal: Time-based windows
        - conversation: Gap detection
        - single: One message per chunk
        - sliding_window: Overlapping windows
        - token_aware: Respect token limits

        Args:
            messages: List of message dictionaries
            strategies: List of strategy names (default: all except single)

        Returns:
            Dictionary mapping strategy names to lists of chunks
        """
        if strategies is None:
            strategies = ["temporal", "conversation", "sliding_window", "token_aware"]

        results = {}

        if "temporal" in strategies:
            results["temporal"] = self.chunk_temporal(messages)

        if "conversation" in strategies:
            results["conversation"] = self.chunk_conversation(messages)

        if "single" in strategies:
            results["single"] = self.chunk_single(messages)

        if "sliding_window" in strategies:
            results["sliding_window"] = self.chunk_sliding_window(messages)

        if "token_aware" in strategies:
            results["token_aware"] = self.chunk_by_tokens(messages)

        # Log summary
        total_chunks = sum(len(chunks) for chunks in results.values())
        self.logger.info(
            f"Chunked {len(messages)} messages into {total_chunks} chunks "
            f"across {len(results)} strategies"
        )

        return results
```

---

### Step 4.10: Add Validation and Statistics Methods

Add to `chunking/service.py`:

```python
    def validate_chunk(self, chunk: Chunk, max_tokens: int = 512) -> bool:
        """
        Validate that a chunk is within token limits.

        Args:
            chunk: Chunk to validate
            max_tokens: Maximum allowed tokens

        Returns:
            True if valid, False otherwise
        """
        token_count = chunk.metadata.get('token_count', 0)
        if token_count == 0:
            # Recalculate if not in metadata
            token_count = self.count_tokens(chunk.content)

        if token_count > max_tokens:
            self.logger.warning(
                f"Chunk {chunk.metadata.get('first_message_id', 'unknown')} "
                f"exceeds {max_tokens} tokens ({token_count} tokens)"
            )
            return False

        return True

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict:
        """
        Get statistics about a list of chunks.

        Useful for debugging and optimization.

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_messages": 0,
                "avg_tokens_per_chunk": 0,
                "max_tokens": 0,
                "min_tokens": 0,
                "avg_messages_per_chunk": 0,
            }

        token_counts = [c.metadata.get('token_count', 0) for c in chunks]
        message_counts = [c.metadata.get('message_count', 0) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "total_messages": sum(message_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "avg_messages_per_chunk": sum(message_counts) / len(chunks) if message_counts else 0,
        }

    def preview_chunks(self, chunks: List[Chunk], max_chunks: int = 5):
        """
        Preview first N chunks for debugging.

        Args:
            chunks: List of chunks to preview
            max_chunks: Maximum number of chunks to show
        """
        for i, chunk in enumerate(chunks[:max_chunks]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Strategy: {chunk.metadata.get('chunk_strategy', 'unknown')}")
            print(f"Messages: {chunk.metadata['message_count']}")
            print(f"Tokens: {chunk.metadata['token_count']}")
            print(f"Content preview: {chunk.content[:200]}...")
```

---

### Step 4.11: Add Convenience Methods

Add to `chunking/service.py`:

```python
    def chunk_from_storage(
        self,
        storage,
        channel_id: str,
        strategies: List[str] = None,
        limit: int = None
    ) -> Dict[str, List[Chunk]]:
        """
        Convenience method to chunk messages directly from storage.

        Args:
            storage: MessageStorage instance
            channel_id: Channel ID to retrieve messages from
            strategies: List of chunking strategies to use
            limit: Optional limit on number of messages to retrieve

        Returns:
            Dictionary mapping strategy names to lists of chunks
        """
        messages = storage.get_channel_messages(channel_id, limit=limit)
        self.logger.info(f"Retrieved {len(messages)} messages from channel {channel_id}")
        return self.chunk_messages(messages, strategies=strategies)
```

---

### Step 4.12: Create Module __init__.py

Create `chunking/__init__.py`:

```python
"""
Chunking module for RAG system.

Provides chunking strategies for Discord messages.
"""

from chunking.base import Chunk
from chunking.service import ChunkingService

__all__ = ["Chunk", "ChunkingService"]
```

---

### Step 4.13: Quick Test Script

Create a test script to verify your implementation:

```python
# test_chunking.py
from storage.messages import MessageStorage
from chunking.service import ChunkingService


def test_chunking():
    """Test chunking implementation with real data."""

    # Get some messages
    storage = MessageStorage()
    messages = storage.get_channel_messages(
        channel_id="YOUR_CHANNEL_ID",  # Replace with real channel ID
        limit=100
    )

    print(f"Retrieved {len(messages)} messages")

    if not messages:
        print("No messages found. Make sure you've stored messages first!")
        return

    # Test all strategies
    chunker = ChunkingService()
    results = chunker.chunk_messages(messages)

    print("\n" + "="*60)
    print("CHUNKING RESULTS")
    print("="*60)

    for strategy, chunks in results.items():
        stats = chunker.get_chunk_statistics(chunks)

        print(f"\n{strategy.upper()}:")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.0f}")
        print(f"  Max tokens: {stats['max_tokens']}")
        print(f"  Min tokens: {stats['min_tokens']}")
        print(f"  Avg messages/chunk: {stats['avg_messages_per_chunk']:.1f}")

        # Validate all chunks
        invalid = [c for c in chunks if not chunker.validate_chunk(c)]
        if invalid:
            print(f"  ⚠️ WARNING: {len(invalid)} chunks exceed 512 token limit!")
        else:
            print(f"  ✅ All chunks within token limits")

    # Preview first chunk of each strategy
    print("\n" + "="*60)
    print("CHUNK PREVIEWS")
    print("="*60)

    for strategy, chunks in results.items():
        if chunks:
            print(f"\n{strategy.upper()} - First Chunk:")
            chunker.preview_chunks(chunks, max_chunks=1)


if __name__ == "__main__":
    test_chunking()
```

**Expected output:**

```
Retrieved 100 messages

============================================================
CHUNKING RESULTS
============================================================

TEMPORAL:
  Chunks: 5
  Avg tokens/chunk: 342
  Max tokens: 489
  Min tokens: 201
  Avg messages/chunk: 20.0
  ✅ All chunks within token limits

CONVERSATION:
  Chunks: 8
  Avg tokens/chunk: 213
  Max tokens: 445
  Min tokens: 98
  Avg messages/chunk: 12.5
  ✅ All chunks within token limits

SLIDING_WINDOW:
  Chunks: 18
  Avg tokens/chunk: 178
  Max tokens: 312
  Min tokens: 87
  Avg messages/chunk: 10.0
  ✅ All chunks within token limits

TOKEN_AWARE:
  Chunks: 6
  Avg tokens/chunk: 487
  Max tokens: 511
  Min tokens: 403
  Avg messages/chunk: 16.7
  ✅ All chunks within token limits

============================================================
CHUNK PREVIEWS
============================================================

TEMPORAL - First Chunk:

--- Chunk 1 ---
Strategy: temporal
Messages: 23
Tokens: 445
Content preview: 2025-01-05 - Alice: Hey everyone!
2025-01-05 - Bob: Hi Alice!
2025-01-05 - Alice: How's it going?...
```

---

## Common Pitfalls - Phase 4

1. **Field name mismatches**: Use `message_id` (not `id`) and `author_name`/`author_display_name` (not `author`) to match database schema
2. **Missing logger**: Always initialize `self.logger = logging.getLogger(__name__)` in `__init__`
3. **Missing config**: Add all `CHUNKING_*` config attributes to `config.py` before using them
4. **Missing dependency**: Add `tiktoken>=0.5.0` to `requirements.txt` and run `pip install tiktoken`
5. **Timestamp parsing**: Handle both ISO formats (with/without Z)
6. **Empty chunks**: Don't create chunks with no messages
7. **Input validation**: Always validate messages list before processing (use `_validate_messages`)
8. **Sorting**: Always sort by timestamp before chunking
9. **Boundary detection**: Time gaps must account for timezone
10. **Metadata**: Include all info needed for filtering later
11. **Token counting**: tiktoken is fast (~10k tokens/sec) but adds ~10% overhead
12. **Overlap validation**: Overlap must be less than window_size
13. **Token limits**: Different embedding models have different max tokens (256-8192)
14. **Tiny chunks**: Set minimum chunk size to avoid single-message chunks (unless intended)
15. **Message retrieval**: Ensure messages from `MessageStorage` use correct field names
16. **All invalid timestamps**: If all messages have invalid timestamps, temporal/conversation chunking will return empty list - fallback to single-message chunking
17. **Unicode handling**: tiktoken handles emojis/special chars correctly, but log if encoding fails
18. **Infinite loops**: In sliding_window, always check `if end == len(messages): break`

---

## Debugging Tips - Phase 4

- **Use chunk statistics**: Call `get_chunk_statistics()` to analyze chunk distribution
- **Validate chunks**: Use `validate_chunk()` before embedding to catch token limit issues
- **Preview chunks**: Use `preview_chunks()` to see what content looks like
- **Print chunk sizes**: See how many messages per chunk
- **Check timestamps**: Verify they're parsed correctly
- **Test boundaries**: Create test data with known gaps
- **Compare strategies**: Visualize chunk boundaries side-by-side
- **Verify token counts**: Log token counts for each chunk (now in metadata)
- **Test overlap**: Verify messages appear in multiple chunks with sliding_window
- **Monitor warnings**: Watch for oversized messages or small chunks
- **Check field names**: Verify messages use `message_id` and `author_name`/`author_display_name`
- **Validate inputs**: Use `_validate_messages()` to catch type errors early
- **Test edge cases**: Empty list, single message, all invalid timestamps
- **Log extensively**: Use different log levels (debug, info, warning) for different scenarios

---

## Performance Considerations - Phase 4

- **Sorting**: O(n log n) complexity, but necessary for temporal coherence
- **Chunk count**: More chunks = more embeddings = more storage costs
- **Content length**: Keep chunks under token limits to avoid truncation
- **Token counting**: tiktoken is fast (~10k tokens/sec) but adds ~10% overhead for large datasets
- **Caching**: Consider caching token counts if re-chunking same messages
- **Strategy selection**:
  - `sliding_window`: More chunks (overlap), better retrieval, more storage
  - `token_aware`: Fewer, optimally-sized chunks, cost-effective
  - `temporal/conversation`: Variable size, depends on data patterns
- **Memory usage**: Large message sets (100k+) may need batch processing
- **Lazy tokenizer loading**: Tokenizer initialized once and reused

---

## Recommended Strategy Combinations

### For Most Use Cases:

1. **Start with**: `token_aware` (respects limits, good baseline)
2. **Add**: `sliding_window` (better retrieval with overlap)
3. **Compare**: Both strategies to see which performs better (Phase 6.5)
4. **Optional**: `conversation` if natural breaks are important

### For Experimentation:

- Try all strategies and compare retrieval quality (Phase 6.5)
- Adjust `window_size`, `overlap`, `max_tokens` parameters
- Measure retrieval metrics (Precision, Recall, MRR)

### Production Recommendation:

```python
# Use token_aware for cost-efficiency
# OR sliding_window for better retrieval
# DON'T use both in production (redundant)

strategies = ["token_aware"]  # Cost-effective
# OR
strategies = ["sliding_window"]  # Better retrieval
```

---

## Key Takeaways

✅ **Token-aware chunking** is essential for production RAG (prevents embedding failures)
✅ **Sliding window with overlap** improves retrieval quality at the cost of more chunks
✅ **Validation and statistics** help debug and optimize chunk strategies
✅ **Field name consistency** prevents bugs (use `message_id`, not `id`)
✅ **Configuration** makes chunking behavior tunable without code changes

**What's Next?**
- Phase 5: Vector Store Abstraction (store these chunks!)
- Phase 6: Multi-Strategy Chunk Storage (compare strategies)
- Phase 6.5: Strategy Evaluation (measure which works best)

---

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)
