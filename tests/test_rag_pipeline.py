import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig, RAGResult
from chunking.constants import ChunkStrategy


@pytest.fixture
def mock_chunks():
    """Sample chunks for testing"""
    return [
        {
            'content': 'We decided to use PostgreSQL for the database',
            'metadata': {
                'timestamp': '2025-01-15 14:23:00',
                'author': 'Alice',
                'channel_id': '123456',
                'message_id': 'msg1',
            },
            'similarity': 0.85
        },
        {
            'content': 'UUID primary keys are better than auto-increment',
            'metadata': {
                'timestamp': '2025-01-15 14:25:00',
                'author': 'Bob',
                'channel_id': '123456',
                'message_id': 'msg2',
            },
            'similarity': 0.75
        },
        {
            'content': 'Not very relevant message about lunch',
            'metadata': {
                'timestamp': '2025-01-15 14:30:00',
                'author': 'Charlie',
                'channel_id': '123456',
                'message_id': 'msg3',
            },
            'similarity': 0.35
        }
    ]


class TestRAGConfig:
    """Test RAGConfig dataclass"""
    
    def test_default_config(self):
        """Should create config with default values"""
        config = RAGConfig()
        
        assert config.top_k == 5
        assert config.similarity_threshold == 0.5
        assert config.max_context_tokens == 4000
        assert config.temperature == 0.7
        assert config.strategy == "tokens"
        assert config.model is None
        assert config.show_sources is False
    
    def test_custom_config(self):
        """Should create config with custom values"""
        config = RAGConfig(
            top_k=10,
            similarity_threshold=0.7,
            max_context_tokens=2000,
            temperature=0.5,
            strategy="single",
            show_sources=True
        )
        
        assert config.top_k == 10
        assert config.similarity_threshold == 0.7
        assert config.max_context_tokens == 2000
        assert config.temperature == 0.5
        assert config.strategy == "single"
        assert config.show_sources is True
    
    def test_invalid_top_k(self):
        """Should raise ValueError for invalid top_k"""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            RAGConfig(top_k=0)
    
    def test_invalid_similarity_threshold_low(self):
        """Should raise ValueError for threshold below 0"""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            RAGConfig(similarity_threshold=-0.1)
    
    def test_invalid_similarity_threshold_high(self):
        """Should raise ValueError for threshold above 1"""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            RAGConfig(similarity_threshold=1.5)
    
    def test_invalid_max_context_tokens(self):
        """Should raise ValueError for too few tokens"""
        with pytest.raises(ValueError, match="max_context_tokens must be at least 100"):
            RAGConfig(max_context_tokens=50)


class TestRAGResult:
    """Test RAGResult dataclass"""
    
    def test_default_result(self):
        """Should create result with minimal data"""
        result = RAGResult(answer="Test answer")
        
        assert result.answer == "Test answer"
        assert result.sources == []
        assert result.config_used is None
        assert result.tokens_used == 0
        assert result.cost == 0.0
        assert result.model == "unknown"
    
    def test_format_for_discord_basic(self):
        """Should format result without sources"""
        result = RAGResult(
            answer="PostgreSQL was chosen",
            model="gpt-4o-mini",
            tokens_used=150,
            cost=0.002
        )
        
        formatted = result.format_for_discord(include_sources=False)
        
        assert "**Answer:**" in formatted
        assert "PostgreSQL was chosen" in formatted
        assert "gpt-4o-mini" in formatted
        assert "150" in formatted
        assert "$0.0020" in formatted
        assert "**Sources:**" not in formatted
    
    def test_format_for_discord_with_sources(self, mock_chunks):
        """Should format result with sources"""
        result = RAGResult(
            answer="PostgreSQL was chosen",
            sources=mock_chunks[:2],
            model="gpt-4o-mini",
            tokens_used=150,
            cost=0.002
        )
        
        formatted = result.format_for_discord(include_sources=True)
        
        assert "**Answer:**" in formatted
        assert "**Sources:**" in formatted
        assert "Alice" in formatted
        assert "Bob" in formatted
        assert "2025-01-15 14:23:00" in formatted
        assert "0.85" in formatted


class TestContextBuilding:
    """Test context building with metadata"""
    
    def test_build_context_with_metadata(self, mock_chunks):
        """Should format chunks with [timestamp] author: content"""
        pipeline = RAGPipeline()
        
        context = pipeline._build_context_with_metadata(
            mock_chunks[:2],  # Just first 2 chunks
            max_tokens=10000  # High limit, no truncation
        )
        
        # Check format
        assert '[2025-01-15 14:23:00] Alice: We decided to use PostgreSQL for the database' in context
        assert '[2025-01-15 14:25:00] Bob: UUID primary keys are better than auto-increment' in context
        
        # Check structure (chunks separated by double newline)
        lines = context.split('\n\n')
        assert len(lines) == 2
    
    def test_build_context_empty_chunks(self):
        """Should handle empty chunks list"""
        pipeline = RAGPipeline()
        
        context = pipeline._build_context_with_metadata([], max_tokens=4000)
        
        assert context == ""
    
    def test_build_context_missing_metadata(self):
        """Should handle chunks with missing metadata"""
        pipeline = RAGPipeline()
        
        chunks = [
            {
                'content': 'Test content',
                'metadata': {},
                'similarity': 0.8
            }
        ]
        
        context = pipeline._build_context_with_metadata(chunks, max_tokens=4000)
        
        # Should use defaults for missing metadata
        assert '[Unknown time] Unknown: Test content' in context
    
    def test_token_limit_truncation(self, mock_chunks):
        """Should truncate context at token limit"""
        pipeline = RAGPipeline()
        
        # Very low token limit (should fit only 1 chunk)
        context = pipeline._build_context_with_metadata(
            mock_chunks[:2],
            max_tokens=20  # Very low limit to force truncation
        )
        
        # Should only include first chunk
        assert 'Alice' in context
        assert 'Bob' not in context
    
    def test_token_limit_respects_order(self, mock_chunks):
        """Should prioritize chunks in order (most relevant first)"""
        pipeline = RAGPipeline()
        
        # Enough for 2 chunks but not 3
        context = pipeline._build_context_with_metadata(
            mock_chunks,
            max_tokens=50  # Even lower limit to force exclusion of 3rd chunk
        )
        
        # Should include first two (highest similarity)
        assert 'Alice' in context
        assert 'Bob' in context
        # Should NOT include third (lowest similarity)
        assert 'Charlie' not in context


class TestFiltering:
    """Test similarity filtering"""
    
    def test_filter_by_similarity(self, mock_chunks):
        """Should filter out low similarity chunks"""
        pipeline = RAGPipeline()
        
        filtered = pipeline._filter_by_similarity(
            mock_chunks,
            threshold=0.5
        )
        
        # Should keep first 2, filter out 3rd (0.35 < 0.5)
        assert len(filtered) == 2
        assert filtered[0]['similarity'] == 0.85
        assert filtered[1]['similarity'] == 0.75
    
    def test_no_filtering_with_zero_threshold(self, mock_chunks):
        """Should keep all chunks with 0.0 threshold"""
        pipeline = RAGPipeline()
        
        filtered = pipeline._filter_by_similarity(
            mock_chunks,
            threshold=0.0
        )
        
        assert len(filtered) == 3
    
    def test_strict_filtering(self, mock_chunks):
        """Should filter with strict threshold"""
        pipeline = RAGPipeline()
        
        filtered = pipeline._filter_by_similarity(
            mock_chunks,
            threshold=0.8
        )
        
        # Only first chunk meets threshold
        assert len(filtered) == 1
        assert filtered[0]['similarity'] == 0.85
    
    def test_filter_empty_list(self):
        """Should handle empty chunks list"""
        pipeline = RAGPipeline()
        
        filtered = pipeline._filter_by_similarity([], threshold=0.5)
        
        assert filtered == []
    
    def test_filter_missing_similarity(self):
        """Should handle chunks without similarity score"""
        pipeline = RAGPipeline()
        
        chunks = [
            {'content': 'Test', 'metadata': {}}
        ]
        
        filtered = pipeline._filter_by_similarity(chunks, threshold=0.5)
        
        # Should filter out (no similarity = 0.0)
        assert len(filtered) == 0


class TestPromptCreation:
    """Test RAG prompt creation"""
    
    def test_create_rag_prompt(self):
        """Should create properly formatted prompt"""
        pipeline = RAGPipeline()
        
        context = "[2025-01-15] Alice: We chose PostgreSQL"
        question = "What database did we choose?"
        
        prompt = pipeline._create_rag_prompt(question, context)
        
        # Check structure
        assert "helpful" in prompt.lower()
        assert "discord" in prompt.lower()
        assert context.lower() in prompt.lower()
        assert question.lower() in prompt.lower()
        
        # Check that context comes before question
        context_pos = prompt.lower().find(context.lower())
        question_pos = prompt.lower().find(question.lower())
        assert context_pos < question_pos
    
    def test_prompt_contains_instructions(self):
        """Should include clear instructions"""
        pipeline = RAGPipeline()
        
        prompt = pipeline._create_rag_prompt("Test?", "Context")
        
        # Should have instructions to prevent hallucination
        assert "ONLY" in prompt or "only" in prompt
        assert "context" in prompt.lower()


class TestRetrieveChunks:
    """Test chunk retrieval"""
    
    @pytest.mark.asyncio
    async def test_retrieve_chunks_success(self, mock_chunks):
        """Should retrieve chunks from memory service"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        pipeline = RAGPipeline(chunked_memory_service=mock_memory)
        
        config = RAGConfig(top_k=5, strategy="single")
        
        chunks = await pipeline._retrieve_chunks("test query", config)
        
        # Should call search with correct params
        mock_memory.search.assert_called_once()
        call_kwargs = mock_memory.search.call_args[1]
        assert call_kwargs['query'] == "test query"
        assert call_kwargs['top_k'] == 5
        assert call_kwargs['strategy'] == ChunkStrategy.SINGLE
        
        assert chunks == mock_chunks
    
    @pytest.mark.asyncio
    async def test_retrieve_chunks_invalid_strategy(self, mock_chunks):
        """Should fallback to default strategy for invalid strategy"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        pipeline = RAGPipeline(chunked_memory_service=mock_memory)
        
        config = RAGConfig(strategy="invalid_strategy")
        
        chunks = await pipeline._retrieve_chunks("test", config)
        
        # Should use TOKENS as fallback
        call_kwargs = mock_memory.search.call_args[1]
        assert call_kwargs['strategy'] == ChunkStrategy.TOKENS


class TestFullPipeline:
    """Integration tests for complete pipeline"""
    
    @pytest.mark.asyncio
    async def test_answer_question_success(self, mock_chunks):
        """Should successfully answer with mocked dependencies"""
        # Mock ChunkedMemoryService
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        # Mock AIService
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'You decided to use PostgreSQL with UUID primary keys.',
            'tokens_total': 150,
            'cost': 0.002,
            'model': 'gpt-4o-mini',
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai,
        )
        
        config = RAGConfig(
            top_k=5,
            similarity_threshold=0.5,
            temperature=0.7,
        )
        
        result = await pipeline.answer_question(
            "What database did we choose?",
            config
        )
        
        # Verify result structure
        assert isinstance(result, RAGResult)
        assert 'PostgreSQL' in result.answer
        assert len(result.sources) == 2  # 2 chunks above threshold (0.85, 0.75)
        assert result.tokens_used == 150
        assert result.cost == 0.002
        assert result.model == 'gpt-4o-mini'
        assert result.config_used == config
        
        # Verify AI was called
        mock_ai.generate.assert_called_once()
        call_kwargs = mock_ai.generate.call_args[1]
        assert 'prompt' in call_kwargs
        assert 'temperature' in call_kwargs
        assert call_kwargs['temperature'] == 0.7
    
    @pytest.mark.asyncio
    async def test_answer_question_no_results(self):
        """Should handle case with no search results"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=[])
        
        pipeline = RAGPipeline(chunked_memory_service=mock_memory)
        
        result = await pipeline.answer_question("Unrelated question")
        
        assert "couldn't find" in result.answer.lower()
        assert len(result.sources) == 0
        assert result.model == "none"
    
    @pytest.mark.asyncio
    async def test_answer_question_all_filtered(self):
        """Should handle case where all chunks are filtered"""
        # Create chunks that will all be filtered by high threshold
        low_similarity_chunks = [
            {
                'content': 'Low relevance message',
                'metadata': {'timestamp': '2025-01-15', 'author': 'Test'},
                'similarity': 0.3
            }
        ]
        
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=low_similarity_chunks)
        
        pipeline = RAGPipeline(chunked_memory_service=mock_memory)
        
        # High threshold filters all chunks (0.3 < 0.9)
        config = RAGConfig(similarity_threshold=0.9)
        
        result = await pipeline.answer_question("Test")
        
        # With high threshold, no chunks pass filtering
        assert "couldn't find" in result.answer.lower()
        assert len(result.sources) == 0
        assert result.model == "none"
    
    @pytest.mark.asyncio
    async def test_answer_question_with_default_config(self, mock_chunks):
        """Should work with default config"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'Answer',
            'tokens_total': 100,
            'cost': 0.001,
            'model': 'gpt-4o-mini',
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai
        )
        
        # Call without config (should use defaults)
        result = await pipeline.answer_question("Test question")
        
        assert isinstance(result, RAGResult)
        assert result.answer == "Answer"
    
    @pytest.mark.asyncio
    async def test_answer_question_error_handling(self, mock_chunks):
        """Should handle errors gracefully"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        mock_ai = MagicMock()
        # Simulate AI service error
        mock_ai.generate = AsyncMock(side_effect=Exception("API Error"))
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai
        )
        
        result = await pipeline.answer_question("Test")
        
        # Should return error result, not crash
        assert isinstance(result, RAGResult)
        assert "error" in result.answer.lower()
        assert result.model == "error"
    
    @pytest.mark.asyncio
    async def test_context_sent_to_llm(self, mock_chunks):
        """Should send formatted context to LLM"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'Answer',
            'tokens_total': 100,
            'cost': 0.001,
            'model': 'gpt-4o-mini',
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai
        )
        
        await pipeline.answer_question("What was decided?")
        
        # Check the prompt sent to AI
        call_kwargs = mock_ai.generate.call_args[1]
        prompt = call_kwargs['prompt']
        
        # Should contain formatted chunks with metadata
        assert 'Alice' in prompt
        assert 'PostgreSQL' in prompt
        assert '2025-01-15' in prompt
        assert 'What was decided?' in prompt


class TestAuthorFiltering:
    """Test author filtering in search"""
    
    def test_search_filter_to_specific_authors(self):
        """Should only return chunks from specified authors"""
        from storage.chunked_memory import ChunkedMemoryService
        
        chunks = [
            {
                'content': 'Alice message 1',
                'metadata': {'author': 'Alice', 'timestamp': '2025-01-15'},
                'similarity': 0.9
            },
            {
                'content': 'Bob message 1',
                'metadata': {'author': 'Bob', 'timestamp': '2025-01-15'},
                'similarity': 0.85
            },
            {
                'content': 'Alice message 2',
                'metadata': {'author': 'Alice', 'timestamp': '2025-01-15'},
                'similarity': 0.8
            },
            {
                'content': 'Charlie message 1',
                'metadata': {'author': 'Charlie', 'timestamp': '2025-01-15'},
                'similarity': 0.75
            }
        ]
        
        mock_vector_store = MagicMock()
        mock_vector_store.query = MagicMock(return_value={
            'documents': [[c['content'] for c in chunks]],
            'metadatas': [[c['metadata'] for c in chunks]],
            'distances': [[1 - c['similarity'] for c in chunks]]
        })
        
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
        
        service = ChunkedMemoryService(
            vector_store=mock_vector_store,
            embedder=mock_embedder
        )
        
        # Filter to only Alice's messages
        results = service.search("test query", top_k=5, filter_authors=['Alice'])
        
        # Should only return Alice's messages
        assert len(results) == 2
        for result in results:
            assert result['metadata']['author'] == 'Alice'
    
    def test_search_filter_multiple_authors(self):
        """Should return chunks from any of the specified authors"""
        from storage.chunked_memory import ChunkedMemoryService
        
        chunks = [
            {'content': 'Alice msg', 'metadata': {'author': 'Alice'}, 'similarity': 0.9},
            {'content': 'Bob msg', 'metadata': {'author': 'Bob'}, 'similarity': 0.85},
            {'content': 'Charlie msg', 'metadata': {'author': 'Charlie'}, 'similarity': 0.8},
        ]
        
        mock_vector_store = MagicMock()
        mock_vector_store.query = MagicMock(return_value={
            'documents': [[c['content'] for c in chunks]],
            'metadatas': [[c['metadata'] for c in chunks]],
            'distances': [[1 - c['similarity'] for c in chunks]]
        })
        
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
        
        service = ChunkedMemoryService(
            vector_store=mock_vector_store,
            embedder=mock_embedder
        )
        
        # Filter to Alice and Charlie
        results = service.search("test", top_k=5, filter_authors=['Alice', 'Charlie'])
        
        assert len(results) == 2
        authors = [r['metadata']['author'] for r in results]
        assert 'Alice' in authors
        assert 'Charlie' in authors
        assert 'Bob' not in authors
    
    def test_search_filter_case_insensitive(self):
        """Should filter authors case-insensitively"""
        from storage.chunked_memory import ChunkedMemoryService
        
        chunks = [
            {'content': 'msg', 'metadata': {'author': 'Alice'}, 'similarity': 0.9},
            {'content': 'msg', 'metadata': {'author': 'bob'}, 'similarity': 0.85},
        ]
        
        mock_vector_store = MagicMock()
        mock_vector_store.query = MagicMock(return_value={
            'documents': [[c['content'] for c in chunks]],
            'metadatas': [[c['metadata'] for c in chunks]],
            'distances': [[1 - c['similarity'] for c in chunks]]
        })
        
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
        
        service = ChunkedMemoryService(
            vector_store=mock_vector_store,
            embedder=mock_embedder
        )
        
        # Search with different case
        results = service.search("test", top_k=5, filter_authors=['alice', 'BOB'])
        
        # Should match both despite different casing
        assert len(results) == 2


class TestBlacklistFiltering:
    """Test blacklist filtering in search"""
    
    def test_search_excludes_blacklisted_authors(self):
        """Should filter out chunks from blacklisted authors"""
        from storage.chunked_memory import ChunkedMemoryService
        from config import Config
        
        # Mock chunks with blacklisted and non-blacklisted authors
        chunks_with_blacklisted = [
            {
                'content': 'Good message from Alice',
                'metadata': {'author': 'Alice', 'timestamp': '2025-01-15'},
                'similarity': 0.9
            },
            {
                'content': 'Message from blacklisted user',
                'metadata': {'author': 'BlacklistedUser', 'timestamp': '2025-01-15'},
                'similarity': 0.85
            },
            {
                'content': 'Another good message from Bob',
                'metadata': {'author': 'Bob', 'timestamp': '2025-01-15'},
                'similarity': 0.8
            }
        ]
        
        # Temporarily set blacklist
        original_blacklist = Config.BLACKLIST_IDS
        Config.BLACKLIST_IDS = ['BlacklistedUser']
        
        try:
            mock_vector_store = MagicMock()
            mock_vector_store.query = MagicMock(return_value={
                'documents': [[c['content'] for c in chunks_with_blacklisted]],
                'metadatas': [[c['metadata'] for c in chunks_with_blacklisted]],
                'distances': [[1 - c['similarity'] for c in chunks_with_blacklisted]]
            })
            
            mock_embedder = MagicMock()
            mock_embedder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
            
            service = ChunkedMemoryService(
                vector_store=mock_vector_store,
                embedder=mock_embedder
            )
            
            results = service.search("test query", top_k=5, exclude_blacklisted=True)
            
            # Should only include non-blacklisted authors
            assert len(results) == 2
            authors = [r['metadata']['author'] for r in results]
            assert 'Alice' in authors
            assert 'Bob' in authors
            assert 'BlacklistedUser' not in authors
            
        finally:
            # Restore original blacklist
            Config.BLACKLIST_IDS = original_blacklist
    
    def test_search_with_blacklist_disabled(self):
        """Should include all chunks when blacklist filtering is disabled"""
        from storage.chunked_memory import ChunkedMemoryService
        from config import Config
        
        chunks_with_blacklisted = [
            {
                'content': 'Message 1',
                'metadata': {'author': 'Alice', 'timestamp': '2025-01-15'},
                'similarity': 0.9
            },
            {
                'content': 'Message 2',
                'metadata': {'author': 'BlacklistedUser', 'timestamp': '2025-01-15'},
                'similarity': 0.85
            }
        ]
        
        original_blacklist = Config.BLACKLIST_IDS
        Config.BLACKLIST_IDS = ['BlacklistedUser']
        
        try:
            mock_vector_store = MagicMock()
            mock_vector_store.query = MagicMock(return_value={
                'documents': [[c['content'] for c in chunks_with_blacklisted]],
                'metadatas': [[c['metadata'] for c in chunks_with_blacklisted]],
                'distances': [[1 - c['similarity'] for c in chunks_with_blacklisted]]
            })
            
            mock_embedder = MagicMock()
            mock_embedder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
            
            service = ChunkedMemoryService(
                vector_store=mock_vector_store,
                embedder=mock_embedder
            )
            
            results = service.search("test query", top_k=5, exclude_blacklisted=False)
            
            # Should include all chunks including blacklisted
            assert len(results) == 2
            authors = [r['metadata']['author'] for r in results]
            assert 'BlacklistedUser' in authors
            
        finally:
            Config.BLACKLIST_IDS = original_blacklist


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_very_long_content_truncation(self):
        """Should handle very long content that exceeds token limit"""
        # Create chunk with very long content
        long_chunk = {
            'content': 'A' * 50000,  # Very long content
            'metadata': {
                'timestamp': '2025-01-15',
                'author': 'Test',
            },
            'similarity': 0.9
        }
        
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=[long_chunk])
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'Answer',
            'tokens_total': 100,
            'cost': 0.001,
            'model': 'gpt-4o-mini',
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai
        )
        
        config = RAGConfig(max_context_tokens=4000)
        
        result = await pipeline.answer_question("Test")
        
        # Should complete without error
        assert isinstance(result, RAGResult)
        
        # Check that prompt wasn't too long
        call_kwargs = mock_ai.generate.call_args[1]
        prompt = call_kwargs['prompt']
        # 4000 tokens * 4 chars ≈ 16000 chars (rough estimate)
        assert len(prompt) < 20000  # Some buffer for overhead

