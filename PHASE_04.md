# Phase 4: Chunking Strategies

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Chunking Strategies

### Learning Objectives
- Understand different chunking approaches
- Learn when to use each strategy
- Design extensible chunking system
- Practice algorithm design

### Design Principles
- **Strategy Pattern**: Different chunking algorithms
- **Configurable Parameters**: Window sizes, gaps as config
- **Extensibility**: Easy to add new strategies

### Implementation Steps

#### Step 4.1: Design Chunk Data Structure

Create `services/chunking_service.py`:

```python
from typing import List, Dict, Optional
from datetime import datetime
from config import Config

class Chunk:
    """
    Data structure for a chunk.
    
    Learning: Good data modeling is crucial for RAG systems.
    Metadata enables filtering and context preservation.
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
        """Convert chunk to dictionary for storage"""
        return {
            "content": self.content,
            "message_ids": self.message_ids,
            "metadata": self.metadata
        }

class ChunkingService:
    """
    Service for chunking messages using different strategies.
    
    Learning: Strategy pattern allows experimenting with different approaches.
    """
    
    def __init__(
        self,
        temporal_window: int = None,
        conversation_gap: int = None
    ):
        self.temporal_window = temporal_window or Config.CHUNKING_TEMPORAL_WINDOW
        self.conversation_gap = conversation_gap or Config.CHUNKING_CONVERSATION_GAP
```

#### Step 4.2: Implement Temporal Chunking

```python
def chunk_temporal(self, messages: List[Dict]) -> List[Chunk]:
    """
    Group messages by time windows.
    
    Learning: Temporal chunking preserves time-based context.
    Useful for conversations that happen over time.
    """
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
        except Exception:
            # Skip messages with invalid timestamps
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
    
    return chunks
```

#### Step 4.3: Implement Conversation Chunking

```python
def chunk_conversation(self, messages: List[Dict]) -> List[Chunk]:
    """
    Group messages by conversation boundaries.
    
    Learning: Conversation chunking detects natural breaks in dialogue.
    Boundaries: time gaps, channel changes, topic shifts.
    """
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
    
    return chunks
```

#### Step 4.4: Implement Single Message Chunking

```python
def chunk_single(self, messages: List[Dict]) -> List[Chunk]:
    """
    Each message is its own chunk.
    
    Learning: Baseline strategy for comparison.
    Maximum granularity, preserves all individual messages.
    """
    chunks = []
    for message in messages:
        chunks.append(self._create_chunk([message], "single"))
    return chunks

def _create_chunk(self, messages: List[Dict], strategy: str) -> Chunk:
    """Helper to create chunk with metadata"""
    if not messages:
        raise ValueError("Cannot create chunk from empty messages")
    
    # Format content
    content_parts = []
    for msg in messages:
        author = msg.get('author_display_name') or msg.get('author', 'Unknown')
        timestamp = msg.get('timestamp', '')[:10]  # Date only
        content = msg.get('content', '').strip()
        if content:
            content_parts.append(f"{timestamp} - {author}: {content}")
    
    content = "\n".join(content_parts)
    
    # Collect metadata
    message_ids = [str(msg.get('id', '')) for msg in messages if msg.get('id')]
    metadata = {
        "chunk_strategy": strategy,
        "channel_id": messages[0].get('channel_id', ''),
        "message_count": len(messages),
        "first_message_id": message_ids[0] if message_ids else '',
        "last_message_id": message_ids[-1] if message_ids else '',
        "first_timestamp": messages[0].get('timestamp', ''),
        "last_timestamp": messages[-1].get('timestamp', ''),
    }
    
    return Chunk(content, message_ids, metadata)
```

#### Step 4.5: Main Chunking Method

```python
def chunk_messages(
    self, 
    messages: List[Dict], 
    strategies: List[str] = None
) -> Dict[str, List[Chunk]]:
    """
    Generate chunks using specified strategies.
    
    Learning: Returns all strategies at once for comparison.
    """
    if strategies is None:
        strategies = ["temporal", "conversation", "single"]
    
    results = {}
    
    if "temporal" in strategies:
        results["temporal"] = self.chunk_temporal(messages)
    
    if "conversation" in strategies:
        results["conversation"] = self.chunk_conversation(messages)
    
    if "single" in strategies:
        results["single"] = self.chunk_single(messages)
    
    return results
```

### Common Pitfalls - Phase 4

1. **Timestamp parsing**: Handle both ISO formats (with/without Z)
2. **Empty chunks**: Don't create chunks with no messages
3. **Sorting**: Always sort by timestamp before chunking
4. **Boundary detection**: Time gaps must account for timezone
5. **Metadata**: Include all info needed for filtering later

### Debugging Tips - Phase 4

- **Print chunk sizes**: See how many messages per chunk
- **Check timestamps**: Verify they're parsed correctly
- **Test boundaries**: Create test data with known gaps
- **Compare strategies**: Visualize chunk boundaries

### Performance Considerations - Phase 4

- **Sorting**: O(n log n) complexity, but necessary
- **Chunk count**: More chunks = more embeddings = more storage
- **Content length**: Keep chunks under token limits