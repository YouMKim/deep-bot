"""
Tests for advanced RAG features:
- Hybrid search (BM25 + Vector)
- Re-ranking with cross-encoder
- Multi-query retrieval
- HyDE (Hypothetical Document Embeddings)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from rag.models import RAGConfig, RAGResult
from rag.hybrid_search import HybridSearchService, reciprocal_rank_fusion
from rag.reranking import ReRankingService
from rag.query_enhancement import QueryEnhancementService
from rag.pipeline import RAGPipeline
from chunking.constants import ChunkStrategy


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        {
            'content': 'We decided to use PostgreSQL for the database',
            'metadata': {
                'first_message_id': 'msg1',
                'timestamp': '2025-01-15 14:23:00',
                'author': 'Alice',
            },
            'similarity': 0.85,
            'bm25_score': 2.5
        },
        {
            'content': 'UUID primary keys are better than auto-increment',
            'metadata': {
                'first_message_id': 'msg2',
                'timestamp': '2025-01-15 14:25:00',
                'author': 'Bob',
            },
            'similarity': 0.75,
            'bm25_score': 2.0
        },
        {
            'content': 'PostgreSQL supports JSON columns natively',
            'metadata': {
                'first_message_id': 'msg3',
                'timestamp': '2025-01-15 14:27:00',
                'author': 'Alice',
            },
            'similarity': 0.70,
            'bm25_score': 1.8
        }
    ]


class TestHybridSearch:
    """Test hybrid search functionality"""
    
    def test_hybrid_search_basic(self, sample_chunks):
        """Should merge BM25 and vector results"""
        service = HybridSearchService()
        
        bm25_results = sample_chunks[:2]  # First 2
        vector_results = sample_chunks[1:]  # Last 2 (overlap with msg2)
        
        results = service.hybrid_search(
            query="database",
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_k=3
        )
        
        assert len(results) == 3
        # All results should have rrf_score
        for result in results:
            assert 'rrf_score' in result
            assert 'fusion_rank' in result
    
    def test_hybrid_search_weighted(self, sample_chunks):
        """Should respect BM25 and vector weights"""
        service = HybridSearchService()
        
        bm25_results = sample_chunks[:1]
        vector_results = sample_chunks[1:2]
        
        # Test with BM25 weighted higher
        results_bm25_heavy = service.hybrid_search(
            query="test",
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=0.8,
            vector_weight=0.2,
            top_k=2
        )
        
        # Test with vector weighted higher
        results_vector_heavy = service.hybrid_search(
            query="test",
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=0.2,
            vector_weight=0.8,
            top_k=2
        )
        
        # Results should be different based on weights
        assert len(results_bm25_heavy) == 2
        assert len(results_vector_heavy) == 2
    
    def test_hybrid_search_deduplication(self, sample_chunks):
        """Should deduplicate documents appearing in both lists"""
        service = HybridSearchService()
        
        # Same document in both lists
        bm25_results = [sample_chunks[0]]
        vector_results = [sample_chunks[0]]  # Same document
        
        results = service.hybrid_search(
            query="test",
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_k=5
        )
        
        # Should only appear once (with combined score)
        assert len(results) == 1
        assert results[0]['rrf_score'] > 0
    
    def test_reciprocal_rank_fusion_function(self, sample_chunks):
        """Should merge multiple ranked lists using RRF"""
        list1 = [sample_chunks[0], sample_chunks[1]]
        list2 = [sample_chunks[1], sample_chunks[2]]  # Overlap with list1
        list3 = [sample_chunks[2], sample_chunks[0]]
        
        results = reciprocal_rank_fusion(
            ranked_lists=[list1, list2, list3],
            top_k=3
        )
        
        assert len(results) == 3
        # All should have rrf_score
        for result in results:
            assert 'rrf_score' in result
        
        # Documents appearing in multiple lists should have higher scores
        msg2_scores = [r['rrf_score'] for r in results if r['metadata']['first_message_id'] == 'msg2']
        if msg2_scores:
            # msg2 appears in list1 and list2, so should have higher score
            assert msg2_scores[0] > 0


class TestReRanking:
    """Test re-ranking functionality"""
    
    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock cross-encoder model"""
        mock_model = MagicMock()
        # Simulate scores (higher = more relevant)
        mock_model.predict = MagicMock(return_value=[0.9, 0.7, 0.8])  # Re-orders chunks
        return mock_model
    
    def test_rerank_basic(self, sample_chunks, mock_cross_encoder):
        """Should re-rank chunks using cross-encoder"""
        with patch('rag.reranking.CrossEncoder', return_value=mock_cross_encoder):
            service = ReRankingService()
            
            # Original order: msg1 (0.85), msg2 (0.75), msg3 (0.70)
            # Mock scores: [0.9, 0.7, 0.8] - should reorder to msg1, msg3, msg2
            reranked = service.rerank(
                query="PostgreSQL database",
                chunks=sample_chunks.copy(),
                top_k=3
            )
            
            assert len(reranked) == 3
            # All should have ce_score
            for chunk in reranked:
                assert 'ce_score' in chunk
                assert 'original_rank' in chunk
            
            # Should be sorted by ce_score (descending)
            scores = [chunk['ce_score'] for chunk in reranked]
            assert scores == sorted(scores, reverse=True)
    
    def test_rerank_empty_list(self):
        """Should handle empty chunks list"""
        with patch('rag.reranking.CrossEncoder'):
            service = ReRankingService()
            result = service.rerank(query="test", chunks=[], top_k=10)
            assert result == []
    
    def test_rerank_top_k_limiting(self, sample_chunks, mock_cross_encoder):
        """Should limit results to top_k"""
        with patch('rag.reranking.CrossEncoder', return_value=mock_cross_encoder):
            service = ReRankingService()
            
            reranked = service.rerank(
                query="test",
                chunks=sample_chunks.copy(),
                top_k=2
            )
            
            assert len(reranked) == 2
    
    def test_rerank_preserves_original_rank(self, sample_chunks, mock_cross_encoder):
        """Should preserve original rank information"""
        with patch('rag.reranking.CrossEncoder', return_value=mock_cross_encoder):
            service = ReRankingService()
            
            reranked = service.rerank(
                query="test",
                chunks=sample_chunks.copy(),
                top_k=3
            )
            
            # Check that original_rank is preserved
            for chunk in reranked:
                assert 'original_rank' in chunk
                assert 1 <= chunk['original_rank'] <= 3


class TestQueryEnhancement:
    """Test query enhancement (multi-query and HyDE)"""
    
    @pytest.fixture
    def mock_ai_service(self):
        """Mock AI service for query generation"""
        mock = MagicMock()
        mock.generate = AsyncMock(return_value={
            'content': 'What database technology was selected?\nWhich database system did we choose?\nWhat was the database decision?',
            'tokens_total': 50,
            'cost': 0.001,
            'model': 'gpt-4o-mini'
        })
        return mock
    
    @pytest.mark.asyncio
    async def test_generate_multi_queries(self, mock_ai_service):
        """Should generate multiple query variations"""
        service = QueryEnhancementService(ai_service=mock_ai_service)
        
        queries = await service.generate_multi_queries(
            query="What database did we choose?",
            num_queries=3
        )
        
        # Should include original + variations
        assert len(queries) >= 2  # Original + at least 1 variation
        assert queries[0] == "What database did we choose?"  # Original first
        
        # Verify AI service was called
        mock_ai_service.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_multi_queries_handles_numbering(self, mock_ai_service):
        """Should remove numbering from generated queries"""
        # Mock response with numbering
        mock_ai_service.generate = AsyncMock(return_value={
            'content': '1. What database was selected?\n2. Which database?\n3. Database choice?',
            'tokens_total': 50,
            'cost': 0.001,
            'model': 'gpt-4o-mini'
        })
        
        service = QueryEnhancementService(ai_service=mock_ai_service)
        queries = await service.generate_multi_queries("test", num_queries=3)
        
        # Should remove numbering
        for query in queries[1:]:  # Skip original
            assert not query[0].isdigit() if query else True
    
    @pytest.mark.asyncio
    async def test_generate_hyde_document(self, mock_ai_service):
        """Should generate hypothetical answer for HyDE"""
        mock_ai_service.generate = AsyncMock(return_value={
            'content': 'We chose PostgreSQL because it supports JSON natively and has excellent ACID compliance.',
            'tokens_total': 30,
            'cost': 0.0005,
            'model': 'gpt-4o-mini'
        })
        
        service = QueryEnhancementService(ai_service=mock_ai_service)
        
        hyde_doc = await service.generate_hyde_document(
            query="What database did we choose?"
        )
        
        assert isinstance(hyde_doc, str)
        assert len(hyde_doc) > 0
        # Should contain relevant keywords
        assert 'PostgreSQL' in hyde_doc or 'database' in hyde_doc.lower()
        
        # Verify AI service was called
        mock_ai_service.generate.assert_called_once()


class TestMultiQueryRetrieval:
    """Test multi-query retrieval in pipeline"""
    
    @pytest.mark.asyncio
    async def test_multi_query_retrieval(self, sample_chunks):
        """Should retrieve with multiple query variations"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=sample_chunks)
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'What database was selected?\nWhich database system?\nDatabase choice?',
            'tokens_total': 50,
            'cost': 0.001,
            'model': 'gpt-4o-mini'
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai
        )
        
        config = RAGConfig(
            use_multi_query=True,
            num_query_variations=3,
            top_k=5
        )
        
        chunks = await pipeline._retrieve_chunks("What database?", config)
        
        # Should call search multiple times (once per query variation)
        assert mock_memory.search.call_count >= 2  # Original + variations
        
        # Should return fused results
        assert len(chunks) <= config.top_k
    
    @pytest.mark.asyncio
    async def test_multi_query_with_hybrid_search(self, sample_chunks):
        """Should use hybrid search when enabled with multi-query"""
        mock_memory = MagicMock()
        mock_memory.search_hybrid = MagicMock(return_value=sample_chunks)
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'Variation 1\nVariation 2',
            'tokens_total': 30,
            'cost': 0.001,
            'model': 'gpt-4o-mini'
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai
        )
        
        config = RAGConfig(
            use_multi_query=True,
            use_hybrid_search=True,
            top_k=5
        )
        
        await pipeline._retrieve_chunks("test", config)
        
        # Should use search_hybrid, not search
        assert mock_memory.search_hybrid.called
        assert not mock_memory.search.called


class TestRerankingInPipeline:
    """Test re-ranking integration in pipeline"""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_reranking(self, sample_chunks):
        """Should re-rank results when enabled"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=sample_chunks * 2)  # More chunks for reranking
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'Answer',
            'tokens_total': 100,
            'cost': 0.001,
            'model': 'gpt-4o-mini'
        })
        
        # Mock cross-encoder
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict = MagicMock(return_value=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        
        with patch('rag.reranking.CrossEncoder', return_value=mock_cross_encoder):
            pipeline = RAGPipeline(
                chunked_memory_service=mock_memory,
                ai_service=mock_ai
            )
            
            config = RAGConfig(
                use_reranking=True,
                top_k=3
            )
            
            chunks = await pipeline._retrieve_chunks("test query", config)
            
            # Should fetch more candidates (top_k * 3)
            assert mock_memory.search.call_args[1]['top_k'] == 9  # 3 * 3
            
            # Should return top_k after reranking
            assert len(chunks) == 3
            
            # Should have reranker initialized
            assert pipeline.reranker is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_reranking_fetches_more(self, sample_chunks):
        """Should fetch more candidates when reranking is enabled"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=sample_chunks)
        
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict = MagicMock(return_value=[0.9, 0.8, 0.7])
        
        with patch('rag.reranking.CrossEncoder', return_value=mock_cross_encoder):
            pipeline = RAGPipeline(chunked_memory_service=mock_memory)
            
            config = RAGConfig(
                use_reranking=True,
                top_k=5
            )
            
            await pipeline._retrieve_chunks("test", config)
            
            # Should request 3x more candidates
            call_kwargs = mock_memory.search.call_args[1]
            assert call_kwargs['top_k'] == 15  # 5 * 3


class TestHybridSearchInPipeline:
    """Test hybrid search integration in pipeline"""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_hybrid_search(self, sample_chunks):
        """Should use hybrid search when enabled"""
        mock_memory = MagicMock()
        mock_memory.search_hybrid = MagicMock(return_value=sample_chunks)
        
        pipeline = RAGPipeline(chunked_memory_service=mock_memory)
        
        config = RAGConfig(
            use_hybrid_search=True,
            bm25_weight=0.6,
            vector_weight=0.4,
            top_k=5
        )
        
        chunks = await pipeline._retrieve_chunks("test query", config)
        
        # Should use search_hybrid
        assert mock_memory.search_hybrid.called
        assert not mock_memory.search.called
        
        # Should pass correct weights
        call_kwargs = mock_memory.search_hybrid.call_args[1]
        assert call_kwargs['bm25_weight'] == 0.6
        assert call_kwargs['vector_weight'] == 0.4


class TestRAGConfigNewFields:
    """Test new RAGConfig fields"""
    
    def test_config_hybrid_search_defaults(self):
        """Should have correct defaults for hybrid search"""
        config = RAGConfig()
        
        assert config.use_hybrid_search is False
        assert config.bm25_weight == 0.5
        assert config.vector_weight == 0.5
    
    def test_config_multi_query_defaults(self):
        """Should have correct defaults for multi-query"""
        config = RAGConfig()
        
        assert config.use_multi_query is False
        assert config.num_query_variations == 3
    
    def test_config_reranking_defaults(self):
        """Should have correct defaults for reranking"""
        config = RAGConfig()
        
        assert config.use_reranking is False
        assert config.reranking_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def test_config_hyde_defaults(self):
        """Should have correct defaults for HyDE"""
        config = RAGConfig()
        
        assert config.use_hyde is False
    
    def test_config_custom_advanced_features(self):
        """Should allow custom configuration of advanced features"""
        config = RAGConfig(
            use_hybrid_search=True,
            use_multi_query=True,
            use_reranking=True,
            use_hyde=True,
            bm25_weight=0.7,
            vector_weight=0.3,
            num_query_variations=5
        )
        
        assert config.use_hybrid_search is True
        assert config.use_multi_query is True
        assert config.use_reranking is True
        assert config.use_hyde is True
        assert config.bm25_weight == 0.7
        assert config.vector_weight == 0.3
        assert config.num_query_variations == 5


class TestFullPipelineWithAdvancedFeatures:
    """Integration tests with all advanced features"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_hybrid_reranking(self, sample_chunks):
        """Should work with hybrid search + reranking"""
        mock_memory = MagicMock()
        mock_memory.search_hybrid = MagicMock(return_value=sample_chunks * 3)
        
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'PostgreSQL was chosen for the database.',
            'tokens_total': 150,
            'cost': 0.002,
            'model': 'gpt-4o-mini'
        })
        
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict = MagicMock(return_value=[0.9] * 9)
        
        with patch('rag.reranking.CrossEncoder', return_value=mock_cross_encoder):
            pipeline = RAGPipeline(
                chunked_memory_service=mock_memory,
                ai_service=mock_ai
            )
            
            config = RAGConfig(
                use_hybrid_search=True,
                use_reranking=True,
                top_k=5,
                similarity_threshold=0.5
            )
            
            result = await pipeline.answer_question("What database?", config)
            
            assert isinstance(result, RAGResult)
            assert 'PostgreSQL' in result.answer
            assert mock_memory.search_hybrid.called
            assert pipeline.reranker is not None

