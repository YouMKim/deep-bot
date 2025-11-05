# Phase 9: Configuration & Polish

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Configuration & Polish

### Learning Objectives
- Learn configuration management
- Practice feature flags
- Understand validation patterns

### Implementation Steps

#### Step 9.1: Add All Config Options

Update `config.py`:

```python
# Embedding Configuration
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")

# Vector Store Configuration
VECTOR_STORE_PROVIDER: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")

# Chunking Configuration
CHUNKING_TEMPORAL_WINDOW: int = int(os.getenv("CHUNKING_TEMPORAL_WINDOW", "300"))
CHUNKING_CONVERSATION_GAP: int = int(os.getenv("CHUNKING_CONVERSATION_GAP", "1800"))

# Rate Limiting
MESSAGE_FETCH_DELAY: float = float(os.getenv("MESSAGE_FETCH_DELAY", "1.0"))
MESSAGE_FETCH_BATCH_SIZE: int = int(os.getenv("MESSAGE_FETCH_BATCH_SIZE", "100"))
MESSAGE_FETCH_PROGRESS_INTERVAL: int = int(os.getenv("MESSAGE_FETCH_PROGRESS_INTERVAL", "100"))
MESSAGE_FETCH_MAX_RETRIES: int = int(os.getenv("MESSAGE_FETCH_MAX_RETRIES", "5"))

# Storage Configuration
RAW_MESSAGES_DIR: str = os.getenv("RAW_MESSAGES_DIR", "data/raw_messages")

# Features
SUMMARY_USE_STORED_MESSAGES: bool = os.getenv(
    "SUMMARY_USE_STORED_MESSAGES", "True"
).lower() == "true"

@classmethod
def validate(cls) -> bool:
    """Validate configuration"""
    required = ["DISCORD_TOKEN"]
    missing = [var for var in required if not getattr(cls, var)]
    
    if missing:
        print(f"❌ Missing required config: {', '.join(missing)}")
        return False
    
    # Validate rate limits
    if cls.MESSAGE_FETCH_DELAY < 0.1:
        print("⚠️ Warning: Rate limit delay too low, may get rate limited")
    
    # Validate chunking
    if cls.CHUNKING_TEMPORAL_WINDOW < 60:
        print("⚠️ Warning: Temporal window too small (< 60s)")
    
    # Validate embedding provider
    if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
        print("⚠️ Warning: OpenAI embedding provider selected but no API key")
    
    return True
```

---

## Quick Reference Checklists

### Phase 1 Checklist
- [ ] Database schema created
- [ ] MessageStorage class implemented
- [ ] Context managers for connections
- [ ] Checkpoint system working
- [ ] Tested with 100 messages
- [ ] Verified indexes work

### Phase 2 Checklist
- [ ] Rate limiting configured
- [ ] Exponential backoff implemented
- [ ] Progress callbacks working
- [ ] Error handling complete
- [ ] Tested with 1000 messages
- [ ] No rate limit errors

### Phase 3 Checklist
- [ ] Abstract base class created
- [ ] SentenceTransformer implemented
- [ ] OpenAI provider implemented
- [ ] Factory pattern working
- [ ] Both providers tested
- [ ] Dimensions verified

### Phase 4 Checklist
- [ ] Chunk data structure designed
- [ ] Temporal chunking implemented
- [ ] Conversation chunking implemented
- [ ] Single-message chunking implemented
- [ ] All strategies tested
- [ ] Metadata preserved

### Phase 5 Checklist
- [ ] Vector store abstract class created
- [ ] ChromaDB adapter implemented
- [ ] Factory pattern working
- [ ] Collections created correctly
- [ ] Queries working
- [ ] Error handling complete

### Phase 6 Checklist
- [ ] ChunkedMemoryService created
- [ ] Store all strategies working
- [ ] Search functionality working
- [ ] Stats collection working
- [ ] Strategy switching works
- [ ] End-to-end tested

### Phase 7 Checklist
- [ ] Bot commands integrated
- [ ] Progress updates working
- [ ] Error messages user-friendly
- [ ] All commands tested
- [ ] Owner checks in place
- [ ] Documentation complete

### Phase 8 Checklist
- [ ] Summary uses stored messages
- [ ] Gap detection working
- [ ] Fallback to API works
- [ ] Performance improved
- [ ] No regressions

### Phase 9 Checklist
- [ ] All config options added
- [ ] Validation working
- [ ] Feature flags working
- [ ] Environment variables documented
- [ ] Defaults are sensible

---

## Common Pitfalls & Debugging

### Database Issues

**Problem**: "database is locked"
- **Solution**: Use context managers, ensure connections are closed
- **Debug**: Check for long-running transactions

**Problem**: Slow queries
- **Solution**: Verify indexes are created, use EXPLAIN QUERY PLAN
- **Debug**: Check if queries are using indexes

### Rate Limiting Issues

**Problem**: Getting 429 errors
- **Solution**: Increase MESSAGE_FETCH_DELAY to 1.5-2.0 seconds
- **Debug**: Log actual request rate

**Problem**: Fetching too slow
- **Solution**: Balance between speed and safety (1.0s is good)
- **Debug**: Monitor progress callbacks

### Embedding Issues

**Problem**: Dimension mismatch
- **Solution**: Verify provider dimensions match collection
- **Debug**: Log embedding dimensions

**Problem**: API errors
- **Solution**: Add retry logic, check API keys
- **Debug**: Test with single encode first

### Chunking Issues

**Problem**: Empty chunks
- **Solution**: Check timestamp parsing, filter empty messages
- **Debug**: Print chunk contents

**Problem**: Too many/few chunks
- **Solution**: Adjust window sizes and gaps
- **Debug**: Visualize chunk boundaries

### Vector Store Issues

**Problem**: Collection not found
- **Solution**: Handle missing collections gracefully
- **Debug**: List all collections

**Problem**: Query returns nothing
- **Solution**: Verify documents were stored, check query format
- **Debug**: Query with known documents

---

## Testing Strategy

### For Each Phase:
1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test components together
3. **Performance Tests**: Measure with real data sizes
4. **Comparison Tests**: Compare strategies side-by-side

### Test Data Sizes:
- Phase 1-2: 100 messages (quick testing)
- Phase 3-4: 1,000 messages (medium testing)
- Phase 5-6: 10,000 messages (stress testing)

---

## Key Design Patterns Summary

1. **Strategy Pattern**: Embedding providers, chunking strategies
2. **Adapter Pattern**: Vector store wrappers
3. **Factory Pattern**: Creating providers from config
4. **Observer Pattern**: Progress callbacks
5. **Repository Pattern**: Message storage abstraction
6. **Dependency Injection**: Pass dependencies to services

---

## Next Steps

Start with **Phase 1** and work through sequentially. Each phase builds on the previous one. After completing each phase:

1. Test thoroughly
2. Understand the design decisions
3. Experiment with parameters
4. Document what you learned
5. Move to next phase

Good luck with your learning journey! 🚀