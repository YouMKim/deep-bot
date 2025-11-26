from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime
import logging
import re
from chunking.base import Chunk
from bot.utils.tokenizer import count_tokens as shared_count_tokens

if TYPE_CHECKING:
    from config import Config


#TODO: for now only testing single chunking 
#TODO: Evaluate how in real performance the other chunking methods perform 
#TODO: Evaluate how much of a problem the token size is -> probably need to just go with bigger modle?


class ChunkingService:
    """
    Service for chunking messages using different strategies.

    Available strategies:
    - temporal: Time-based windows (may exceed token limits)
    - conversation: Gap detection (may exceed token limits)
    - single: One message per chunk
    - sliding_window: Overlapping windows 
    - author: Group by author
    - tokens: Token-aware chunking (respects token limits)
    
    Note: Use the 'tokens' strategy if you need to guarantee chunks stay within token limits.
    The 'temporal' and 'conversation' strategies may create chunks that exceed limits.
    """
    
    def __init__(
        self,
        temporal_window: int = None,
        conversation_gap: int = None,
        config: Optional['Config'] = None
    ):
        from config import Config as ConfigClass
        
        self.logger = logging.getLogger(__name__)
        self.config = config or ConfigClass
        self.temporal_window = temporal_window or self.config.CHUNKING_TEMPORAL_WINDOW
        self.conversation_gap = conversation_gap or self.config.CHUNKING_CONVERSATION_GAP 

    def _validate_messages(self, messages: List[Dict]) -> bool:
        if not isinstance(messages, list):
            raise ValueError(f"Messages must be a list, got {type(messages)}")
        #empty list is empty chunk 
        if not messages:
            return True 
        
        sample = messages[0]
        if not isinstance(sample,dict):
            raise ValueError(f"Message must be a list of dictionaries, got {type(sample)}")
        return True 

    def count_tokens(self, text: str, model: str = "cl100k_base") -> int:
        """
        Count tokens in text using shared tokenizer utility.
        
        Args:
            text: Text to count tokens for
            model: Encoding model (kept for API compatibility, ignored)
            
        Returns:
            Number of tokens
        """
        return shared_count_tokens(text)

    def chunk_messages(
        self,
        messages: List[Dict],
        strategies: List[str] = ["single"]
    ) -> Dict[str, List[Chunk]]:
        """
        Generate chunks using specified strategies.

        Automatically sorts messages by timestamp before chunking.
        Returns all strategies at once for comparison.

        Available strategies:
        - temporal: Time-based windows
        - conversation: Gap detection
        - single: One message per chunk
        - sliding_window: Overlapping windows
        - author: Group by author
        - tokens: token aware chunking

        Args:
            messages: List of message dictionaries (will be sorted by timestamp)
            strategies: List of strategy names (default: ["single"])

        Returns:
            Dictionary mapping strategy names to lists of chunks
        """
        self._validate_messages(messages)
        if not messages:
            return {}



        results = {}

        if "temporal" in strategies:
            results["temporal"] = self.chunk_temporal(messages)

        if "conversation" in strategies:
            results["conversation"] = self.chunk_conversation(messages)

        if "single" in strategies:
            results["single"] = self.chunk_single(messages)

        if "sliding_window" in strategies:
            results["sliding_window"] = self.chunk_sliding_window(messages)

        if "author" in strategies:
            results["author"] = self.chunk_by_author(messages)
        
        if "tokens" in strategies:
            results["tokens"] = self.chunk_by_tokens(messages)

        total_chunks = sum(len(chunks) for chunks in results.values())
        self.logger.info(
            f"Chunked {len(messages)} messages into {total_chunks} chunks "
            f"across {len(results)} strategies"
        )

        return results

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
            token_count = self.count_tokens(chunk.content)

        if token_count > max_tokens:
            self.logger.warning(
                f"Chunk {chunk.metadata.get('first_message_id', 'unknown')} "
                f"exceeds {max_tokens} tokens ({token_count} tokens)"
            )
            return False

        return True

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict:
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
        for i, chunk in enumerate(chunks[:max_chunks]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Strategy: {chunk.metadata.get('chunk_strategy', 'unknown')}")
            print(f"Messages: {chunk.metadata.get('message_count', 0)}")
            print(f"Tokens: {chunk.metadata.get('token_count', 0)}")
            print(f"Authors: {chunk.metadata.get('author_count', 0)}")
            print(f"Content preview: {chunk.content[:200]}...")

    def chunk_single(self, messages: List[Dict]) -> List[Chunk]:
        """
        Each message is its own chunk.
        Baseline strategy for comparison.
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

    def _split_message_by_tokens(
        self,
        message: Dict,
        max_tokens: int
    ) -> List[Dict]:
        """
        Split a single message that exceeds token limit into smaller parts.
        
        Strategy:
        1. Try splitting by sentences (., !, ?)
        2. If still too large, split by words
        3. Preserve message metadata for each split
        
        Args:
            message: Message dictionary to split
            max_tokens: Maximum tokens per split
            
        Returns:
            List of message dictionaries (split parts)
        """
        author = message.get('author_display_name') or message.get('author_name') or 'Unknown'
        timestamp = message.get('timestamp', '')[:10]
        content = message.get('content', '').strip()
        
        if not content:
            return []
        
        sentences = re.split(r'([.!?]+\s+|$)', content)
        sentence_parts = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_parts.append(sentences[i] + sentences[i + 1])
            else:
                sentence_parts.append(sentences[i])
        
        sentence_parts = [s.strip() for s in sentence_parts if s.strip()]
        
        if not sentence_parts:
            words = content.split()
            sentence_parts = [' '.join(words[i:i+50]) for i in range(0, len(words), 50)]
        
        splits = []
        current_part = []
        current_tokens = 0
        
        for sentence in sentence_parts:
            test_content = '\n'.join(current_part + [sentence]) if current_part else sentence
            formatted = f"{timestamp} - {author}: {test_content}\n"
            part_tokens = self.count_tokens(formatted)
            
            if part_tokens > max_tokens and current_part:
                part_content = '\n'.join(current_part)
                splits.append({
                    **message,
                    'content': part_content,
                    'message_id': f"{message.get('message_id', 'unknown')}_part{len(splits) + 1}",
                    'is_split': True,
                    'split_index': len(splits) + 1
                })
                current_part = [sentence]
                current_tokens = self.count_tokens(f"{timestamp} - {author}: {sentence}\n")
            elif part_tokens > max_tokens:
                words = sentence.split()
                word_chunks = []
                current_words = []
                
                for word in words:
                    test_words = ' '.join(current_words + [word])
                    word_tokens = self.count_tokens(f"{timestamp} - {author}: {test_words}\n")
                    
                    if word_tokens > max_tokens and current_words:
                        word_chunks.append(' '.join(current_words))
                        current_words = [word]
                    else:
                        current_words.append(word)
                
                if current_words:
                    word_chunks.append(' '.join(current_words))
                
                for i, word_chunk in enumerate(word_chunks):
                    splits.append({
                        **message,
                        'content': word_chunk,
                        'message_id': f"{message.get('message_id', 'unknown')}_part{len(splits) + 1}",
                        'is_split': True,
                        'split_index': len(splits) + 1
                    })
            else:
                current_part.append(sentence)
                current_tokens = part_tokens
        
        if current_part:
            part_content = '\n'.join(current_part)
            splits.append({
                **message,
                'content': part_content,
                'message_id': f"{message.get('message_id', 'unknown')}_part{len(splits) + 1}",
                'is_split': True,
                'split_index': len(splits) + 1
            })
        
        return splits

    def chunk_by_tokens(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        min_chunk_size: int = None
    ) -> List[Chunk]:
        """
        Create chunks respecting token limits.
        
        If a single message exceeds the token limit, it will be split into
        smaller parts rather than skipped.

        - all-MiniLM-L6-v2: 512 tokens but realistically probably want 200 
        - OpenAI ada-002: 8192 tokens

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

        max_tokens = max_tokens or self.config.CHUNKING_MAX_TOKENS
        min_chunk_size = min_chunk_size or self.config.CHUNKING_MIN_CHUNK_SIZE

        sorted_messages = sorted(messages, key=lambda m: m.get('timestamp', ''))
        chunks = []
        current_chunk = []
        current_tokens = 0

        for message in sorted_messages:
            author = message.get('author_display_name') or message.get('author_name') or 'Unknown'
            timestamp = message.get('timestamp', '')[:10]
            content = message.get('content', '').strip()

            if not content:
                continue

            formatted = f"{timestamp} - {author}: {content}\n"
            msg_tokens = self.count_tokens(formatted)

            # If single message exceeds limit, split it
            if msg_tokens > max_tokens:
                # Save current chunk if it exists
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, "token_aware"))
                    current_chunk = []
                    current_tokens = 0
                elif current_chunk:
                    # Still save even if below min size
                    chunks.append(self._create_chunk(current_chunk, "token_aware"))
                    current_chunk = []
                    current_tokens = 0

                # Split the oversized message
                self.logger.warning(
                    f"Message {message.get('message_id', 'unknown')} exceeds {max_tokens} tokens "
                    f"({msg_tokens} tokens). Splitting into smaller parts."
                )
                
                message_splits = self._split_message_by_tokens(message, max_tokens)
                
                # Process each split as a separate message
                for split_msg in message_splits:
                    split_author = split_msg.get('author_display_name') or split_msg.get('author_name') or 'Unknown'
                    split_timestamp = split_msg.get('timestamp', '')[:10]
                    split_content = split_msg.get('content', '').strip()
                    
                    split_formatted = f"{split_timestamp} - {split_author}: {split_content}\n"
                    split_tokens = self.count_tokens(split_formatted)
                    
                    # Check if split can be added to current chunk
                    if current_tokens + split_tokens > max_tokens:
                        # Save current chunk
                        if current_chunk and len(current_chunk) >= min_chunk_size:
                            chunks.append(self._create_chunk(current_chunk, "token_aware"))
                        elif current_chunk:
                            chunks.append(self._create_chunk(current_chunk, "token_aware"))
                        
                        # Start new chunk with split
                        current_chunk = [split_msg]
                        current_tokens = split_tokens
                    else:
                        # Add split to current chunk
                        current_chunk.append(split_msg)
                        current_tokens += split_tokens
                
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

    
    def chunk_temporal(self, sorted_messages: List[Dict]) -> List[Chunk]:
        """
        Group messages by time windows.

        !!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!
        MESSAGES MUST BE SORTED BY TIMESTAMP BEFORE CHUNKING
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Example:
            Window = 1 hour
            - Chunk 1: Messages from 10:00-11:00
            - Chunk 2: Messages from 11:00-12:00
            
        Args:
            messages: List of message dictionaries
        Returns:
            List of temporal chunks
        """
        self._validate_messages(sorted_messages)
        if not sorted_messages:
            return []

        chunks = []
        current_chunk = []
        window_start = None

        for message in sorted_messages:
            try:
                timestamp_str = message.get('timestamp', '')
                if 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                self.logger.warning(f"Skipping message with invalid timestamp: {e}")
                continue

            if window_start is None:
                window_start = timestamp

            time_diff = (timestamp - window_start).total_seconds()

            if time_diff > self.temporal_window:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, "temporal"))
                current_chunk = [message]
                window_start = timestamp
            else:
                current_chunk.append(message)

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, "temporal"))

        self.logger.info(f"Temporal chunking created {len(chunks)} chunks from {len(sorted_messages)} messages")
        return chunks


    def chunk_conversation(self, sorted_messages: List[Dict]) -> List[Chunk]:
        """
        Group messages by conversation boundaries.

        !!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!
        MESSAGES MUST BE SORTED BY TIMESTAMP BEFORE CHUNKING
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Example:
            Gap = 30 minutes
            - 10:00 AM: "Hello"
            - 10:01 AM: "Hi there"
            - 10:35 AM: "Different topic" <- NEW CHUNK (gap > 30min)

        Args:
            sorted_messages: List of message dictionaries (MUST be sorted by timestamp)

        Returns:
            List of conversation chunks
        """
        self._validate_messages(sorted_messages)
        if not sorted_messages:
            return []

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
            except Exception as e:
                self.logger.warning(f"Skipping message with invalid timestamp: {e}")
                continue

            channel_id = message.get('channel_id')
            
            # Check for conversation boundary
            is_boundary = False

            if last_timestamp:
                time_gap = (timestamp - last_timestamp).total_seconds()
                if time_gap > self.conversation_gap:
                    is_boundary = True

            if is_boundary and current_chunk:
                chunks.append(self._create_chunk(current_chunk, "conversation"))
                current_chunk = [message]
            else:
                current_chunk.append(message)

            last_timestamp = timestamp

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, "conversation"))

        self.logger.info(f"Conversation chunking created {len(chunks)} chunks from {len(sorted_messages)} messages")
        return chunks

    

    def chunk_sliding_window(
        self,
        sorted_messages: List[Dict],
        window_size: int = None,
        overlap: int = None
    ) -> List[Chunk]:
        """
        Create overlapping chunks with sliding window.

        Example (window=3, overlap=1):
            Chunk 1: [msg1, msg2, msg3]
            Chunk 2: [msg3, msg4, msg5]  # msg3 overlaps
            Chunk 3: [msg5, msg6, msg7]  # msg5 overlaps

        Args:
            sorted_messages: List of messages to chunk (MUST be sorted by timestamp)
            window_size: Number of messages per chunk (default from config)
            overlap: Number of messages to overlap between chunks (default from config)

        Returns:
            List of chunks with overlapping messages
        """
        self._validate_messages(sorted_messages)
        if not sorted_messages:
            return []

        window_size = window_size or self.config.CHUNKING_WINDOW_SIZE
        overlap = overlap or self.config.CHUNKING_OVERLAP

        if overlap >= window_size:
            self.logger.warning(
                f"Overlap ({overlap}) >= window_size ({window_size}), "
                f"setting overlap to {window_size - 1}"
            )
            overlap = max(1, window_size - 1)

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
            f"from {len(sorted_messages)} messages (window={window_size}, overlap={overlap})"
        )
        return chunks

    def chunk_by_author(
        self,
        sorted_messages: List[Dict],
        max_gap_seconds: int = 600
    ) -> List[Chunk]:
        """
        Group consecutive messages from the same author.
        
        Useful for Discord where users often send multiple messages in sequence.
        Creates a new chunk when:
        - Author changes
        - Time gap exceeds max_gap_seconds (default: 10 minutes)
        
        Example:
            - User A: "NARUTO IS THE BEST ANIME OF ALL TIME"
            - User A: "YOU HAVE SUCH TRASH TASTES"  <- Same chunk
            - User B: "WTF ONE PIECE IS WAY BETTER"     <- NEW CHUNK (different author)
            - User A: "RASENGAN"       <- NEW CHUNK (author changed back)
        
        Args:
            sorted_messages: List of messages sorted by timestamp
            max_gap_seconds: Maximum time gap between messages from same author (default: 600 = 10 min)
            
        Returns:
            List of author-based chunks
        """
        self._validate_messages(sorted_messages)
        if not sorted_messages:
            return []
        
        chunks = []
        current_chunk = []
        last_author_id = None
        last_timestamp = None
        
        for message in sorted_messages:
            try:
                timestamp_str = message.get('timestamp', '')
                if 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                self.logger.warning(f"Skipping message with invalid timestamp: {e}")
                continue
            
            author_id = message.get('author_id') or message.get('author_name')
            
            # Check if we should start a new chunk
            is_new_chunk = False
            
            # Different author = new chunk
            if last_author_id and author_id != last_author_id:
                is_new_chunk = True
            
            # Time gap too large = new chunk (even if same author)
            if last_timestamp and author_id == last_author_id:
                time_gap = (timestamp - last_timestamp).total_seconds()
                if time_gap > max_gap_seconds:
                    is_new_chunk = True
            
            if is_new_chunk and current_chunk:
                chunks.append(self._create_chunk(current_chunk, "author"))
                current_chunk = [message]
            else:
                current_chunk.append(message)
            
            last_author_id = author_id
            last_timestamp = timestamp
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, "author"))
        
        self.logger.info(
            f"Author chunking created {len(chunks)} chunks from {len(sorted_messages)} messages"
        )
        return chunks



    def _create_chunk(self, messages: List[Dict], strategy: str) -> Chunk:
        """
        Helper to create chunk with metadata and validation.
        
        Enhanced metadata includes:
        - All unique authors in the chunk
        - Author count
        - Primary author (most messages)
        
        Metadata is formatted to be ChromaDB-compatible:
        - Lists are stored as comma-separated strings
        - All values are strings, numbers, or booleans

        Args:
            messages: List of messages to include in chunk
            strategy: Name of chunking strategy used

        Returns:
            Chunk object with content, IDs, and metadata
        """
        if not messages:
            raise ValueError("Cannot create chunk from empty messages")

        content_parts = []
        authors = {}  
        author_ids = set()
        author_names = {}  
        
        for msg in messages:
            author_name = msg.get('author_display_name') or msg.get('author_name') or 'Unknown'
            author_id = msg.get('author_id') or msg.get('author_name') or 'unknown'
            
            authors[author_id] = authors.get(author_id, 0) + 1
            author_ids.add(author_id)
            if author_id not in author_names:
                author_names[author_id] = author_name
            
            timestamp = msg.get('timestamp', '')[:10] 
            content = msg.get('content', '').strip()
            if content:
                content_parts.append(f"{timestamp} - {author_name}: {content}")

        content = "\n".join(content_parts)

        # Validate token count
        token_count = self.count_tokens(content)

        # Collect message IDs
        message_ids = [str(msg.get('message_id', '')) for msg in messages if msg.get('message_id')]
        
        # Find primary author (author with most messages)
        primary_author_id = max(authors.items(), key=lambda x: x[1])[0] if authors else None
        primary_author_name = author_names.get(primary_author_id) if primary_author_id else None
        
        metadata = {
            "chunk_strategy": strategy,
            "channel_id": messages[0].get('channel_id', ''),
            "message_count": len(messages),
            "token_count": token_count,
            
            # Author information (ChromaDB-compatible)
            "author_count": len(author_ids), 
            "authors": ",".join(sorted(author_ids)),  
            "primary_author_id": primary_author_id or '', 
            "primary_author_name": primary_author_name or '',
            
            "author_distribution": ",".join([f"{aid}:{count}" for aid, count in sorted(authors.items())]),

            "first_message_id": message_ids[0] if message_ids else '',
            "last_message_id": message_ids[-1] if message_ids else '',
            "first_timestamp": messages[0].get('timestamp', ''),
            "last_timestamp": messages[-1].get('timestamp', ''),
        }

        if token_count > 512:
            self.logger.warning(
                f"Chunk exceeds 512 tokens ({token_count}). "
                f"May fail with some embedding models."
            )

        if len(messages) == 1:
            self.logger.debug(f"Single-message chunk created (strategy: {strategy})")

        return Chunk(content, message_ids, metadata)

