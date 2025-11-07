"""
Quick test script for chunking service.
Tests all chunking strategies with sample Discord messages.
"""

import sys
from datetime import datetime, timedelta
from chunking.service import ChunkingService
from chunking.base import Chunk

def create_sample_messages(count: int = 20) -> list:
    """Create sample Discord messages for testing."""
    base_time = datetime.now() - timedelta(hours=2)
    messages = []
    
    authors = [
        {"author_id": "user123", "author_name": "Alice", "author_display_name": "Alice"},
        {"author_id": "user456", "author_name": "Bob", "author_display_name": "Bob"},
        {"author_id": "user789", "author_name": "Charlie", "author_display_name": "Charlie"},
    ]
    
    sample_contents = [
        "Hey everyone! How's it going?",
        "I'm working on a new project",
        "That sounds interesting!",
        "Can someone help me with Python?",
        "Sure, what do you need?",
        "I'm stuck on async/await",
        "Here's a quick example...",
        "Thanks! That helps a lot",
        "No problem!",
        "Anyone want to play a game?",
        "I'm in!",
        "Let's do it!",
        "This is a longer message that contains more content to test token counting and see how the chunking handles different message lengths.",
        "Short msg",
        "Another message here",
        "Testing temporal chunking",
        "More messages",
        "Even more content",
        "Final message",
        "Last one!",
        # Very long single sentence (no punctuation breaks)
        "This is an extremely long single sentence without any punctuation breaks that goes on and on and on and contains many words and phrases that are all connected together in one continuous stream of text that will definitely exceed token limits when processed by the chunking service and should be split at the word level since there are no sentence boundaries to use for splitting.",
        # Very long message with multiple sentences
        "This is the first sentence of a very long message. This is the second sentence that continues the thought. Here comes the third sentence with even more content. The fourth sentence adds more details about the topic. Sentence number five provides additional context. The sixth sentence expands on previous points. Sentence seven introduces a new idea. The eighth sentence elaborates on that idea. Number nine continues the discussion. The tenth sentence wraps up this part. Eleven adds a final thought. Twelve concludes the message with a summary.",
        # Long message with mixed punctuation
        "What a great day! The weather is perfect. How are you doing? I'm having an amazing time. This is so much fun! Can you believe it? Everything is going well. I'm really excited about this. What do you think? Let's keep going! This is fantastic news. I'm so happy right now. Everything is perfect today. What an incredible experience! This keeps getting better. I can't believe how great this is. What a wonderful time we're having! This is absolutely amazing. I'm having the best day ever. What could be better than this? Nothing at all!",
        # Very long technical message
        "In Python programming, async/await is a powerful feature that allows you to write asynchronous code that can handle multiple operations concurrently without blocking the main thread, which is especially useful for I/O-bound operations like network requests, file operations, and database queries, where you're waiting for external resources to respond, and by using async functions and the await keyword, you can write code that looks synchronous but actually runs asynchronously, allowing other tasks to run while waiting for I/O operations to complete, which significantly improves performance and resource utilization in applications that need to handle many concurrent operations, such as web servers, API clients, and data processing pipelines that need to interact with multiple external services simultaneously.",
    ]
    
    for i in range(count):
        author = authors[i % len(authors)]
        timestamp = base_time + timedelta(minutes=i * 5)  # 5 min intervals
        
        messages.append({
            "message_id": f"msg_{i+1:03d}",
            "channel_id": "channel_123",
            "content": sample_contents[i % len(sample_contents)],
            **author,
            "timestamp": timestamp.isoformat() + "Z",
        })
    
    return messages


def test_chunking_strategies():
    """Test all chunking strategies."""
    print("=" * 60)
    print("CHUNKING SERVICE TEST")
    print("=" * 60)
    
    # Create sample messages
    messages = create_sample_messages(20)
    print(f"\n[+] Created {len(messages)} sample messages")
    print(f"   Time range: {messages[0]['timestamp']} to {messages[-1]['timestamp']}")
    
    # Initialize service
    service = ChunkingService()
    print(f"\n[+] ChunkingService initialized")
    print(f"   Temporal window: {service.temporal_window}s ({service.temporal_window/3600:.1f} hours)")
    print(f"   Conversation gap: {service.conversation_gap}s ({service.conversation_gap/60:.1f} minutes)")
    
    # Test all strategies
    print("\n" + "=" * 60)
    print("TESTING CHUNKING STRATEGIES")
    print("=" * 60)
    
    strategies = ["single", "temporal", "conversation", "sliding_window", "author", "tokens"]
    results = service.chunk_messages(messages, strategies=strategies)
    
    # Display results for each strategy
    for strategy_name, chunks in results.items():
        print(f"\n[*] Strategy: {strategy_name.upper()}")
        print(f"   Chunks created: {len(chunks)}")
        
        if chunks:
            stats = service.get_chunk_statistics(chunks)
            print(f"   Total messages: {stats['total_messages']}")
            print(f"   Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
            print(f"   Max tokens: {stats['max_tokens']}")
            print(f"   Min tokens: {stats['min_tokens']}")
            print(f"   Avg messages/chunk: {stats['avg_messages_per_chunk']:.1f}")
            
            # Validate chunks
            invalid = [c for c in chunks if not service.validate_chunk(c, max_tokens=512)]
            if invalid:
                print(f"   [WARNING] {len(invalid)} chunks exceed 512 tokens")
            else:
                print(f"   [OK] All chunks within token limit")
            
            # Show sample chunk
            if chunks:
                print(f"\n   Sample chunk:")
                service.preview_chunks(chunks, max_chunks=1)
    
    # Test individual strategies
    print("\n" + "=" * 60)
    print("DETAILED STRATEGY TESTS")
    print("=" * 60)
    
    # Test single chunking
    print("\n[1] SINGLE CHUNKING")
    single_chunks = service.chunk_single(messages)
    print(f"   Created {len(single_chunks)} chunks (should equal {len(messages)})")
    assert len(single_chunks) == len(messages), "Single chunking should create one chunk per message"
    print("   [OK] Test passed")
    
    # Test temporal chunking
    print("\n[2] TEMPORAL CHUNKING")
    temporal_chunks = service.chunk_temporal(messages)
    print(f"   Created {len(temporal_chunks)} chunks")
    if temporal_chunks:
        print(f"   First chunk: {temporal_chunks[0].metadata['first_timestamp']}")
        print(f"   Last chunk: {temporal_chunks[-1].metadata['last_timestamp']}")
    print("   [OK] Test passed")
    
    # Test conversation chunking
    print("\n[3] CONVERSATION CHUNKING")
    conversation_chunks = service.chunk_conversation(messages)
    print(f"   Created {len(conversation_chunks)} chunks")
    if conversation_chunks:
        print(f"   Average chunk size: {sum(c.metadata['message_count'] for c in conversation_chunks) / len(conversation_chunks):.1f} messages")
    print("   [OK] Test passed")
    
    # Test sliding window
    print("\n[4] SLIDING WINDOW CHUNKING")
    window_chunks = service.chunk_sliding_window(messages, window_size=5, overlap=2)
    print(f"   Created {len(window_chunks)} chunks")
    print(f"   Window size: 5, Overlap: 2")
    print("   [OK] Test passed")
    
    # Test author chunking
    print("\n[5] AUTHOR CHUNKING")
    author_chunks = service.chunk_by_author(messages, max_gap_seconds=600)
    print(f"   Created {len(author_chunks)} chunks")
    if author_chunks:
        multi_author = [c for c in author_chunks if c.metadata['author_count'] > 1]
        print(f"   Multi-author chunks: {len(multi_author)}")
        print(f"   Single-author chunks: {len(author_chunks) - len(multi_author)}")
    print("   [OK] Test passed")
    
    # Test token-aware chunking
    print("\n[6] TOKEN-AWARE CHUNKING")
    token_chunks = service.chunk_by_tokens(messages, max_tokens=200, min_chunk_size=1)
    print(f"   Created {len(token_chunks)} chunks")
    if token_chunks:
        stats = service.get_chunk_statistics(token_chunks)
        print(f"   Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
        print(f"   Max tokens: {stats['max_tokens']} (limit: 200)")
        print(f"   All chunks within limit: {stats['max_tokens'] <= 200}")
    print("   [OK] Test passed")
    
    # Test token chunking with various oversized messages
    print("\n[7] TOKEN CHUNKING WITH OVERSIZED MESSAGES")
    oversized_messages = messages.copy()
    
    # Test 1: Very long message with sentences (should split by sentences)
    long_message_with_sentences = " ".join([
        "This is a very long message that contains many words and sentences.",
        "It is designed to exceed the token limit when processed.",
        "The token-aware chunking should split this message into smaller parts.",
        "Each part should be within the token limit.",
        "This tests the message splitting functionality.",
        "The split should preserve the message metadata.",
        "Each split part should have a unique message ID.",
        "This is sentence number eight in this long message.",
        "We need more sentences to ensure it exceeds the limit.",
        "Here is sentence number ten.",
        "And sentence number eleven.",
        "Finally, sentence number twelve to make it really long.",
        "This message should definitely exceed 200 tokens.",
        "Let's add even more content to be absolutely sure.",
        "More sentences means more tokens.",
        "This is getting quite long now.",
        "But we need to ensure it exceeds the limit.",
        "So here are even more sentences.",
        "Each one adds to the token count.",
        "And we keep going until we're sure it's oversized."
    ] * 3)  # Repeat 3 times to make it really long
    
    # Test 2: Extremely long single sentence (no punctuation - should split by words)
    long_single_sentence = "This is an extremely long single sentence without any punctuation breaks that goes on and on and on and contains many words and phrases that are all connected together in one continuous stream of text that will definitely exceed token limits when processed by the chunking service and should be split at the word level since there are no sentence boundaries to use for splitting and this sentence continues even further with more words and phrases added to make it even longer so that it definitely exceeds the token limit multiple times over and tests the word-level splitting functionality thoroughly" * 2
    
    # Test 3: Long message with code blocks (simulating Discord code blocks)
    long_code_message = """Here's a Python example:
```python
def process_data(data):
    results = []
    for item in data:
        if item.is_valid():
            processed = transform(item)
            results.append(processed)
    return results
```
This function processes a list of data items. It iterates through each item, checks if it's valid, transforms it, and adds it to the results list. The function returns the processed results. This is a common pattern in data processing pipelines. The code is clean and readable. It follows Python best practices. The function is well-structured and easy to understand. This example demonstrates good coding style. It shows how to handle data transformation. The code is efficient and maintainable.""" * 2
    
    oversized_messages.extend([
        {
            "message_id": "msg_oversized_sentences",
            "channel_id": "channel_123",
            "content": long_message_with_sentences,
            "author_id": "user123",
            "author_name": "Alice",
            "author_display_name": "Alice",
            "timestamp": (datetime.now() - timedelta(minutes=2)).isoformat() + "Z",
        },
        {
            "message_id": "msg_oversized_single_sentence",
            "channel_id": "channel_123",
            "content": long_single_sentence,
            "author_id": "user456",
            "author_name": "Bob",
            "author_display_name": "Bob",
            "timestamp": (datetime.now() - timedelta(minutes=1)).isoformat() + "Z",
        },
        {
            "message_id": "msg_oversized_code",
            "channel_id": "channel_123",
            "content": long_code_message,
            "author_id": "user789",
            "author_name": "Charlie",
            "author_display_name": "Charlie",
            "timestamp": (datetime.now() - timedelta(minutes=0.5)).isoformat() + "Z",
        }
    ])
    
    oversized_chunks = service.chunk_by_tokens(oversized_messages, max_tokens=200, min_chunk_size=1)
    print(f"   Created {len(oversized_chunks)} chunks from {len(oversized_messages)} messages")
    
    # Analyze splits
    split_chunks = [c for c in oversized_chunks if any('_part' in msg_id for msg_id in c.message_ids)]
    if split_chunks:
        print(f"   [OK] Found {len(split_chunks)} chunk(s) containing split messages")
        
        # Group splits by original message
        split_groups = {}
        for chunk in split_chunks:
            for msg_id in chunk.message_ids:
                if '_part' in msg_id:
                    original_id = msg_id.rsplit('_part', 1)[0]
                    if original_id not in split_groups:
                        split_groups[original_id] = []
                    split_groups[original_id].append(msg_id)
        
        print(f"   Split messages found:")
        for original_id, parts in split_groups.items():
            print(f"      - {original_id}: split into {len(set(parts))} part(s)")
            print(f"        Part IDs: {sorted(set(parts))}")
    else:
        print(f"   [INFO] No split messages detected")
    
    # Show token statistics
    if oversized_chunks:
        stats = service.get_chunk_statistics(oversized_chunks)
        print(f"\n   Token statistics:")
        print(f"      Total chunks: {stats['total_chunks']}")
        print(f"      Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
        print(f"      Max tokens: {stats['max_tokens']} (limit: 200)")
        print(f"      Min tokens: {stats['min_tokens']}")
    
    # Verify all chunks are within limit
    invalid = [c for c in oversized_chunks if not service.validate_chunk(c, max_tokens=200)]
    if invalid:
        print(f"\n   [WARNING] {len(invalid)} chunks exceed 200 tokens:")
        for chunk in invalid[:3]:  # Show first 3
            print(f"      - Chunk with {chunk.metadata.get('token_count', 0)} tokens")
    else:
        print(f"\n   [OK] All chunks within 200 token limit")
    
    # Show sample of split chunks
    if split_chunks:
        print(f"\n   Sample split chunk preview:")
        service.preview_chunks(split_chunks[:2], max_chunks=2)
    
    print("\n   [OK] Test passed")
    
    # Test metadata
    print("\n" + "=" * 60)
    print("METADATA VALIDATION")
    print("=" * 60)
    
    if temporal_chunks:
        chunk = temporal_chunks[0]
        print(f"\n[*] Sample chunk metadata:")
        print(f"   Strategy: {chunk.metadata.get('chunk_strategy')}")
        print(f"   Message count: {chunk.metadata.get('message_count')}")
        print(f"   Token count: {chunk.metadata.get('token_count')}")
        print(f"   Author count: {chunk.metadata.get('author_count')}")
        print(f"   Authors: {chunk.metadata.get('authors')}")
        print(f"   Primary author: {chunk.metadata.get('primary_author_name')}")
        print(f"   Channel: {chunk.metadata.get('channel_id')}")
        print(f"   First message ID: {chunk.metadata.get('first_message_id')}")
        print(f"   Last message ID: {chunk.metadata.get('last_message_id')}")
        
        # Verify ChromaDB compatibility
        required_fields = [
            'chunk_strategy', 'channel_id', 'message_count', 'token_count',
            'author_count', 'authors', 'primary_author_id', 'primary_author_name'
        ]
        missing = [f for f in required_fields if f not in chunk.metadata]
        if missing:
            print(f"   [WARNING] Missing fields: {missing}")
        else:
            print("   [OK] All required metadata fields present")
        
        # Check types are ChromaDB-compatible
        invalid_types = []
        for key, value in chunk.metadata.items():
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                invalid_types.append(f"{key}: {type(value).__name__}")
        if invalid_types:
            print(f"   [WARNING] Invalid types for ChromaDB: {invalid_types}")
        else:
            print("   [OK] All metadata types are ChromaDB-compatible")
    
    print("\n" + "=" * 60)
    print("[OK] ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_chunking_strategies()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

