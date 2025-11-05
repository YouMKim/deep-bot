# Phase 16: Advanced RAG Techniques

## Overview

In **Phases 14-15**, we learned hybrid search, reranking, and query optimization - the fundamental building blocks of modern RAG systems. Now we'll explore **cutting-edge RAG techniques** from recent research that push the boundaries of retrieval quality and generation accuracy.

**Learning Objectives:**
- Implement HyDE (Hypothetical Document Embeddings) for improved retrieval
- Build Self-RAG system with reflection and self-evaluation
- Create RAG Fusion for multi-query synthesis
- Understand iterative retrieval and adaptive RAG
- Compare advanced techniques to see when each is most effective

**Prerequisites:** Phases 14-15 (Hybrid Search, Reranking)

**Estimated Time:** 6-8 hours

---

## Table of Contents

1. [HyDE: Hypothetical Document Embeddings](#1-hyde-hypothetical-document-embeddings)
2. [Self-RAG: Self-Reflective Retrieval](#2-self-rag-self-reflective-retrieval)
3. [RAG Fusion: Multi-Query Synthesis](#3-rag-fusion-multi-query-synthesis)
4. [Iterative Retrieval](#4-iterative-retrieval)
5. [Adaptive RAG](#5-adaptive-rag)
6. [Discord Commands](#6-discord-commands)
7. [Comparison & Evaluation](#7-comparison--evaluation)

---

## 1. HyDE: Hypothetical Document Embeddings

### The Core Idea

**Problem:**
- Query: "What causes rain?"
- Query is a **question**, but documents contain **answers**
- Semantic gap: questions and answers live in different embedding spaces

**HyDE Solution:**
```
User Query: "What causes rain?"
      ↓
LLM generates hypothetical answer:
"Rain is caused by water vapor condensing in clouds..."
      ↓
Embed the ANSWER (not the query)
      ↓
Search with answer embedding (finds similar answers in docs)
```

**Why It Works:**
- Answer-to-answer similarity > question-to-answer similarity
- LLM creates "ideal" answer in the same semantic space as real docs
- Even if hypothetical answer is wrong, it's in the right "neighborhood"

### Implementation

Create `services/hyde_service.py`:

```python
from typing import List, Dict
from openai import OpenAI
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
import config
import logging

logger = logging.getLogger(__name__)

class HyDEService:
    """
    HyDE: Hypothetical Document Embeddings

    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()

    def generate_hypothetical_document(
        self,
        query: str,
        num_documents: int = 1,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate hypothetical answer(s) to the query.

        Args:
            query: User question
            num_documents: Number of hypothetical docs to generate
            temperature: LLM temperature (higher = more diverse)

        Returns:
            List of hypothetical documents
        """
        prompt = f"""Given this question, write a detailed answer that would be found in a knowledge base or documentation.

Question: {query}

Write a comprehensive answer (2-3 paragraphs) that directly addresses this question. Include specific details, examples, and explanations."""

        hypothetical_docs = []

        for _ in range(num_documents):
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=300
            )

            hypothetical_doc = response.choices[0].message.content.strip()
            hypothetical_docs.append(hypothetical_doc)

        return hypothetical_docs

    async def search_with_hyde(
        self,
        query: str,
        top_k: int = 10,
        num_hypothetical_docs: int = 1
    ) -> Dict:
        """
        Search using HyDE approach.

        Steps:
        1. Generate hypothetical document(s)
        2. Embed hypothetical document(s)
        3. Search with hypothetical embeddings
        4. (Optional) Fuse results if multiple hypothetical docs

        Returns:
            {
                "query": str,
                "hypothetical_docs": List[str],
                "results": List[Dict]
            }
        """
        # Step 1: Generate hypothetical documents
        logger.info(f"Generating {num_hypothetical_docs} hypothetical document(s)")
        hypothetical_docs = self.generate_hypothetical_document(
            query,
            num_documents=num_hypothetical_docs,
            temperature=0.7
        )

        # Step 2 & 3: Embed and search with each hypothetical doc
        all_results = []

        for hyp_doc in hypothetical_docs:
            # Embed hypothetical document
            embedding = await self.embedding_service.embed_text(hyp_doc)

            # Search with hypothetical embedding
            results = await self.vector_db.search(
                embedding=embedding,
                top_k=top_k
            )
            all_results.append(results)

        # Step 4: Fuse results if multiple hypothetical docs
        if num_hypothetical_docs > 1:
            # Use RRF to combine
            final_results = self._fuse_results(all_results, top_k)
        else:
            final_results = all_results[0]

        return {
            "query": query,
            "hypothetical_docs": hypothetical_docs,
            "results": final_results
        }

    def _fuse_results(
        self,
        results_list: List[List[Dict]],
        top_k: int
    ) -> List[Dict]:
        """Fuse results from multiple hypothetical documents using RRF."""
        from collections import defaultdict

        doc_scores = defaultdict(float)
        doc_map = {}
        k = 60

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get("id")
                doc_scores[doc_id] += 1 / (k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {**doc_map[doc_id], "hyde_score": score}
            for doc_id, score in sorted_docs[:top_k]
        ]

    async def compare_hyde_vs_normal(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Compare HyDE search vs normal query embedding.

        Returns:
            {
                "query": str,
                "normal_search": List[Dict],
                "hyde_search": List[Dict],
                "hypothetical_doc": str,
                "overlap": int
            }
        """
        # Normal search (embed query directly)
        query_embedding = await self.embedding_service.embed_text(query)
        normal_results = await self.vector_db.search(
            embedding=query_embedding,
            top_k=top_k
        )

        # HyDE search
        hyde_result = await self.search_with_hyde(query, top_k=top_k)

        # Calculate overlap
        normal_ids = {doc["id"] for doc in normal_results}
        hyde_ids = {doc["id"] for doc in hyde_result["results"]}
        overlap = len(normal_ids & hyde_ids)

        return {
            "query": query,
            "normal_search": normal_results,
            "hyde_search": hyde_result["results"],
            "hypothetical_doc": hyde_result["hypothetical_docs"][0],
            "overlap": overlap,
            "overlap_pct": (overlap / top_k) * 100
        }
```

### When to Use HyDE

✅ **Use HyDE when:**
- Users ask **questions** but your docs contain **answers**
- High semantic gap between query and documents
- You have access to a strong LLM for generation
- Precision is more important than recall

❌ **Don't use HyDE when:**
- Keyword matching is important (e.g., searching for specific names/codes)
- Documents are question-style (FAQ databases)
- Very low latency required (HyDE adds LLM call overhead)

---

## 2. Self-RAG: Self-Reflective Retrieval

### The Core Idea

**Problem:**
- Traditional RAG always retrieves, even when not needed
- No self-evaluation of retrieved context quality
- Can't detect when generation goes off-track

**Self-RAG Solution:**
```
User Query → [DECISION 1: Do we need retrieval?]
              ↓ (Yes)
         Retrieve docs → [DECISION 2: Are docs relevant?]
                         ↓ (Yes)
                    Generate answer → [DECISION 3: Is answer supported?]
                                      ↓ (Yes)
                                 Return answer
```

**Three Reflection Points:**
1. **Retrieval Decision**: "Do I need external knowledge?"
2. **Relevance Check**: "Are retrieved docs actually relevant?"
3. **Support Check**: "Is my answer grounded in the retrieved docs?"

### Implementation

Create `services/self_rag_service.py`:

```python
from typing import List, Dict, Tuple
from openai import OpenAI
from services.vector_db_service import VectorDBService
from services.embedding_service import EmbeddingService
import config
import logging

logger = logging.getLogger(__name__)

class SelfRAGService:
    """
    Self-RAG: Self-Reflective Retrieval-Augmented Generation

    Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
    https://arxiv.org/abs/2310.11511
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_db = VectorDBService()
        self.embedding_service = EmbeddingService()

    async def decide_retrieval_needed(self, query: str) -> Tuple[bool, str]:
        """
        Reflection 1: Decide if retrieval is needed.

        Returns:
            (needs_retrieval: bool, reasoning: str)
        """
        prompt = f"""Given this user query, decide if external knowledge retrieval is needed.

Query: "{query}"

Answer with:
- "YES" if the query requires specific facts, data, or context not in general knowledge
- "NO" if it's a general question that can be answered without external sources

Format:
Decision: [YES/NO]
Reasoning: [1-2 sentences]

Examples:
Query: "What is 2+2?"
Decision: NO
Reasoning: This is basic arithmetic that doesn't require external knowledge.

Query: "What trips did we take in 2023?"
Decision: YES
Reasoning: This requires specific factual information from conversation history.

Now analyze the query above:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )

        result = response.choices[0].message.content.strip()

        # Parse response
        needs_retrieval = "YES" in result.upper().split("\n")[0]
        reasoning = result.split("Reasoning:")[-1].strip() if "Reasoning:" in result else ""

        return needs_retrieval, reasoning

    async def evaluate_relevance(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Tuple[Dict, float, str]]:
        """
        Reflection 2: Evaluate relevance of retrieved documents.

        Returns:
            List of (document, relevance_score, reasoning)
        """
        if not documents:
            return []

        evaluated_docs = []

        for doc in documents:
            prompt = f"""Evaluate if this document is relevant to answering the query.

Query: "{query}"

Document:
{doc.get('content', '')[:500]}

Rate relevance on scale 1-10 and explain why.

Format:
Score: [1-10]
Reasoning: [1-2 sentences]"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )

            result = response.choices[0].message.content.strip()

            # Parse score
            try:
                score_line = [line for line in result.split("\n") if "Score:" in line][0]
                score = float(score_line.split(":")[-1].strip().split()[0])
            except:
                score = 5.0

            reasoning = result.split("Reasoning:")[-1].strip() if "Reasoning:" in result else ""

            evaluated_docs.append((doc, score / 10, reasoning))

        # Sort by relevance score
        evaluated_docs.sort(key=lambda x: x[1], reverse=True)

        return evaluated_docs

    async def evaluate_answer_support(
        self,
        query: str,
        answer: str,
        context_docs: List[Dict]
    ) -> Tuple[bool, float, str]:
        """
        Reflection 3: Check if answer is supported by retrieved context.

        Returns:
            (is_supported: bool, confidence: float, reasoning: str)
        """
        context_text = "\n\n".join([
            f"Doc {i+1}: {doc.get('content', '')[:300]}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""Evaluate if the generated answer is supported by the provided context.

Query: "{query}"

Context:
{context_text}

Generated Answer:
{answer}

Evaluate:
1. Is the answer factually grounded in the context?
2. Does it contain hallucinations or unsupported claims?

Format:
Supported: [YES/NO]
Confidence: [0.0-1.0]
Reasoning: [2-3 sentences]"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        result = response.choices[0].message.content.strip()

        # Parse response
        is_supported = "YES" in result.split("\n")[0].upper()

        try:
            confidence_line = [line for line in result.split("\n") if "Confidence:" in line][0]
            confidence = float(confidence_line.split(":")[-1].strip())
        except:
            confidence = 0.5

        reasoning = result.split("Reasoning:")[-1].strip() if "Reasoning:" in result else ""

        return is_supported, confidence, reasoning

    async def self_rag_query(
        self,
        query: str,
        top_k: int = 5,
        relevance_threshold: float = 0.6
    ) -> Dict:
        """
        Complete Self-RAG pipeline with all three reflection points.

        Returns:
            {
                "query": str,
                "reflection_1": {"needs_retrieval": bool, "reasoning": str},
                "reflection_2": {"evaluated_docs": List, "kept_docs": List},
                "answer": str,
                "reflection_3": {"supported": bool, "confidence": float, "reasoning": str},
                "final_confidence": float
            }
        """
        result = {"query": query}

        # Reflection 1: Do we need retrieval?
        needs_retrieval, retrieval_reasoning = await self.decide_retrieval_needed(query)

        result["reflection_1"] = {
            "needs_retrieval": needs_retrieval,
            "reasoning": retrieval_reasoning
        }

        if not needs_retrieval:
            # Generate answer without retrieval
            answer_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )
            answer = answer_response.choices[0].message.content

            result["answer"] = answer
            result["reflection_2"] = None
            result["reflection_3"] = {
                "supported": True,
                "confidence": 0.8,
                "reasoning": "No retrieval needed - general knowledge answer"
            }
            result["final_confidence"] = 0.8

            return result

        # Retrieve documents
        query_embedding = await self.embedding_service.embed_text(query)
        retrieved_docs = await self.vector_db.search(
            embedding=query_embedding,
            top_k=top_k
        )

        # Reflection 2: Are docs relevant?
        evaluated_docs = await self.evaluate_relevance(query, retrieved_docs)

        # Filter by relevance threshold
        kept_docs = [
            (doc, score, reasoning)
            for doc, score, reasoning in evaluated_docs
            if score >= relevance_threshold
        ]

        result["reflection_2"] = {
            "evaluated_docs": [
                {"doc_id": doc["id"], "score": score, "reasoning": reasoning}
                for doc, score, reasoning in evaluated_docs
            ],
            "kept_docs": len(kept_docs),
            "filtered_out": len(evaluated_docs) - len(kept_docs)
        }

        if not kept_docs:
            result["answer"] = "I don't have enough relevant information to answer this question."
            result["reflection_3"] = {
                "supported": False,
                "confidence": 0.0,
                "reasoning": "No relevant documents found"
            }
            result["final_confidence"] = 0.0
            return result

        # Generate answer with kept docs
        context = "\n\n".join([
            f"Source {i+1}:\n{doc.get('content', '')}"
            for i, (doc, _, _) in enumerate(kept_docs)
        ])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Provide a clear, concise answer based only on the information in the context."""

        answer_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        answer = answer_response.choices[0].message.content

        result["answer"] = answer

        # Reflection 3: Is answer supported?
        is_supported, support_confidence, support_reasoning = await self.evaluate_answer_support(
            query,
            answer,
            [doc for doc, _, _ in kept_docs]
        )

        result["reflection_3"] = {
            "supported": is_supported,
            "confidence": support_confidence,
            "reasoning": support_reasoning
        }

        # Calculate final confidence
        avg_doc_relevance = sum(score for _, score, _ in kept_docs) / len(kept_docs)
        final_confidence = (avg_doc_relevance + support_confidence) / 2

        result["final_confidence"] = final_confidence

        return result
```

### Benefits of Self-RAG

1. **Efficiency**: Skips retrieval when not needed (saves time & cost)
2. **Quality Control**: Filters out irrelevant documents
3. **Transparency**: Provides reasoning for each decision
4. **Hallucination Detection**: Catches unsupported claims
5. **Confidence Scores**: Quantifies answer reliability

---

## 3. RAG Fusion: Multi-Query Synthesis

### The Core Idea

**Problem:**
- Single query might miss relevant docs due to phrasing
- User query may be ambiguous or underspecified

**RAG Fusion Solution:**
```
User Query: "best python frameworks"
      ↓
Generate multiple perspectives:
  1. "top Python web frameworks 2024"
  2. "most popular Python libraries"
  3. "recommended Python frameworks for beginners"
      ↓
Search with ALL queries in parallel
      ↓
Fuse results using RRF
      ↓
Rerank fused results
      ↓
Generate comprehensive answer
```

### Implementation

Create `services/rag_fusion_service.py`:

```python
from typing import List, Dict
from openai import OpenAI
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from services.reranking_service import RerankingService
import config
import logging
import asyncio

logger = logging.getLogger(__name__)

class RAGFusionService:
    """
    RAG Fusion: Multi-query retrieval with reciprocal rank fusion.

    Paper: "Forget RAG, the Future is RAG-Fusion"
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()
        self.reranker = RerankingService()

    def generate_multi_query(
        self,
        query: str,
        num_queries: int = 4
    ) -> List[str]:
        """
        Generate multiple query perspectives.

        Args:
            query: Original user query
            num_queries: Number of query variations to generate

        Returns:
            List of query variations (including original)
        """
        prompt = f"""You are a helpful assistant that generates multiple search queries.

Generate {num_queries} different search queries that explore different aspects and perspectives of this question:

Original question: "{query}"

Generate queries that:
1. Rephrase using different vocabulary
2. Break down complex questions into sub-questions
3. Explore related angles and perspectives
4. Are more specific or more general

Return only the queries, one per line, without numbering."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=200
        )

        generated_queries = response.choices[0].message.content.strip().split("\n")
        generated_queries = [q.strip() for q in generated_queries if q.strip()]

        # Include original query
        return [query] + generated_queries[:num_queries]

    async def search_all_queries(
        self,
        queries: List[str],
        top_k: int = 20
    ) -> List[List[Dict]]:
        """
        Search with all queries in parallel.

        Returns:
            List of result lists (one per query)
        """
        async def search_single(query: str):
            embedding = await self.embedding_service.embed_text(query)
            return await self.vector_db.search(embedding=embedding, top_k=top_k)

        # Parallel search
        results = await asyncio.gather(*[search_single(q) for q in queries])

        return results

    def reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict]],
        k: int = 60,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Fuse multiple result lists using RRF.

        Formula: RRF_score(d) = Σ 1/(k + rank(d))

        Args:
            results_list: List of ranked result lists
            k: RRF constant (default 60)
            top_k: Number of final results

        Returns:
            Fused and ranked results
        """
        from collections import defaultdict

        doc_scores = defaultdict(float)
        doc_map = {}

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get("id")
                # RRF formula
                doc_scores[doc_id] += 1.0 / (k + rank)

                # Store doc (take first occurrence)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_items = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {**doc_map[doc_id], "rrf_score": score}
            for doc_id, score in sorted_items[:top_k]
        ]

    async def rag_fusion_search(
        self,
        query: str,
        num_queries: int = 4,
        top_k: int = 10,
        use_reranking: bool = True
    ) -> Dict:
        """
        Complete RAG Fusion pipeline.

        Steps:
        1. Generate multiple query perspectives
        2. Search with all queries in parallel
        3. Fuse results using RRF
        4. (Optional) Rerank fused results
        5. Return top-k

        Returns:
            {
                "original_query": str,
                "generated_queries": List[str],
                "search_results": List[List[Dict]],
                "fused_results": List[Dict],
                "final_results": List[Dict]
            }
        """
        # Step 1: Generate queries
        logger.info(f"Generating {num_queries} query variations")
        queries = self.generate_multi_query(query, num_queries)

        # Step 2: Parallel search
        logger.info(f"Searching with {len(queries)} queries")
        search_results = await self.search_all_queries(queries, top_k=20)

        # Step 3: RRF fusion
        logger.info("Fusing results with RRF")
        fused_results = self.reciprocal_rank_fusion(search_results, top_k=50)

        # Step 4: Optional reranking
        if use_reranking:
            logger.info("Reranking fused results")
            reranked = self.reranker.rerank(query, fused_results, top_k=top_k)
            final_results = [
                {**doc, "rerank_score": score}
                for doc, score in reranked
            ]
        else:
            final_results = fused_results[:top_k]

        return {
            "original_query": query,
            "generated_queries": queries,
            "search_results": search_results,
            "fused_results": fused_results,
            "final_results": final_results
        }

    async def generate_fusion_answer(
        self,
        query: str,
        num_queries: int = 4,
        top_k: int = 5
    ) -> Dict:
        """
        RAG Fusion with answer generation.

        Returns:
            {
                "query": str,
                "generated_queries": List[str],
                "results": List[Dict],
                "answer": str,
                "sources": List[str]
            }
        """
        # Get fused results
        fusion_result = await self.rag_fusion_search(
            query,
            num_queries=num_queries,
            top_k=top_k,
            use_reranking=True
        )

        # Build context from results
        context = "\n\n".join([
            f"Source {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(fusion_result["final_results"])
        ])

        # Generate answer
        prompt = f"""Answer the question based on the provided sources.

Question: {query}

Sources:
{context}

Provide a comprehensive answer that synthesizes information from multiple sources."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        # Extract sources
        sources = [
            f"[{i+1}] {doc.get('author', 'Unknown')} - {doc.get('timestamp', 'Unknown')}"
            for i, doc in enumerate(fusion_result["final_results"])
        ]

        return {
            "query": query,
            "generated_queries": fusion_result["generated_queries"],
            "results": fusion_result["final_results"],
            "answer": answer,
            "sources": sources
        }
```

---

## 4. Iterative Retrieval

### The Concept

Instead of retrieving once, retrieve **iteratively** based on what you learn:

```
Query → Retrieve → Generate partial answer →
        ↓
    [Is answer complete?]
        ↓ NO
    Generate follow-up query → Retrieve more → Update answer
        ↓
    [Is answer complete?]
        ↓ YES
    Return final answer
```

### Simple Implementation

Add to `services/iterative_rag_service.py`:

```python
class IterativeRAGService:
    """Iterative retrieval for complex queries."""

    async def iterative_rag(
        self,
        query: str,
        max_iterations: int = 3,
        top_k: int = 5
    ) -> Dict:
        """
        Iteratively retrieve and refine answer.

        Returns:
            {
                "query": str,
                "iterations": List[Dict],
                "final_answer": str
            }
        """
        iterations = []
        accumulated_context = []
        current_query = query

        for i in range(max_iterations):
            # Retrieve
            embedding = await self.embedding_service.embed_text(current_query)
            results = await self.vector_db.search(embedding=embedding, top_k=top_k)

            accumulated_context.extend(results)

            # Generate partial answer
            context_text = "\n\n".join([
                doc.get("content", "") for doc in accumulated_context[-top_k*2:]
            ])

            prompt = f"""Based on the context, answer this question: {query}

Context:
{context_text}

If you have a complete answer, start with "COMPLETE:"
If you need more information, start with "INCOMPLETE:" and specify what's missing."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )

            answer = response.choices[0].message.content

            iterations.append({
                "iteration": i + 1,
                "query": current_query,
                "results_count": len(results),
                "partial_answer": answer
            })

            # Check if complete
            if answer.startswith("COMPLETE:"):
                break

            # Generate follow-up query
            if answer.startswith("INCOMPLETE:"):
                missing_info = answer.split("INCOMPLETE:")[-1].strip()
                current_query = f"Regarding '{query}', specifically: {missing_info}"

        final_answer = iterations[-1]["partial_answer"].replace("COMPLETE:", "").strip()

        return {
            "query": query,
            "iterations": iterations,
            "final_answer": final_answer,
            "total_iterations": len(iterations)
        }
```

---

## 5. Adaptive RAG

### The Concept

**Choose the right RAG strategy based on query characteristics:**

```python
class AdaptiveRAGService:
    """Adaptively select RAG strategy based on query type."""

    async def classify_query(self, query: str) -> str:
        """
        Classify query type:
        - "simple": Straightforward factual question
        - "complex": Multi-hop reasoning required
        - "exploratory": Broad topic exploration
        - "conversational": Requires context/memory
        """
        # Use LLM or simple rules
        pass

    async def adaptive_search(self, query: str) -> Dict:
        """
        Select best RAG strategy for query.

        Simple → Basic vector search
        Complex → Iterative RAG
        Exploratory → RAG Fusion
        Conversational → Self-RAG with conversation memory
        """
        query_type = await self.classify_query(query)

        if query_type == "simple":
            return await self.basic_search(query)
        elif query_type == "complex":
            return await self.iterative_rag.iterative_rag(query)
        elif query_type == "exploratory":
            return await self.rag_fusion.rag_fusion_search(query)
        else:  # conversational
            return await self.self_rag.self_rag_query(query)
```

---

## 6. Discord Commands

Add to `cogs/advanced_rag_cog.py`:

```python
@commands.command(name="hyde_search")
async def hyde_search(self, ctx, *, query: str):
    """
    Search using HyDE (Hypothetical Document Embeddings).

    Usage: !hyde_search what causes rain?
    """
    await ctx.send(f"🔮 HyDE Search: `{query}`")

    result = await self.hyde_service.search_with_hyde(query, top_k=5)

    embed = discord.Embed(title="🔮 HyDE Search Results", color=discord.Color.purple())

    # Show hypothetical document
    embed.add_field(
        name="Generated Hypothetical Answer",
        value=result["hypothetical_docs"][0][:500] + "...",
        inline=False
    )

    # Show results
    for idx, doc in enumerate(result["results"][:3], 1):
        embed.add_field(
            name=f"{idx}. {doc.get('author', 'Unknown')}",
            value=doc["content"][:150] + "...",
            inline=False
        )

    await ctx.send(embed=embed)

@commands.command(name="self_rag")
async def self_rag(self, ctx, *, query: str):
    """
    Self-reflective RAG with quality checks.

    Usage: !self_rag what is 2+2?
    """
    await ctx.send(f"🤔 Self-RAG: `{query}`")

    result = await self.self_rag_service.self_rag_query(query)

    embed = discord.Embed(title="🤔 Self-RAG Results", color=discord.Color.blue())

    # Reflection 1
    r1 = result["reflection_1"]
    embed.add_field(
        name="1️⃣ Retrieval Decision",
        value=f"{'✅ Needed' if r1['needs_retrieval'] else '❌ Not needed'}\n{r1['reasoning']}",
        inline=False
    )

    # Reflection 2
    if result["reflection_2"]:
        r2 = result["reflection_2"]
        embed.add_field(
            name="2️⃣ Relevance Check",
            value=f"Kept: {r2['kept_docs']} | Filtered: {r2['filtered_out']}",
            inline=False
        )

    # Answer
    embed.add_field(
        name="📝 Answer",
        value=result["answer"],
        inline=False
    )

    # Reflection 3
    r3 = result["reflection_3"]
    embed.add_field(
        name="3️⃣ Support Check",
        value=f"Supported: {'✅' if r3['supported'] else '❌'} | "
              f"Confidence: {r3['confidence']:.2f}\n{r3['reasoning']}",
        inline=False
    )

    # Final confidence
    embed.set_footer(text=f"Final Confidence: {result['final_confidence']:.2f}")

    await ctx.send(embed=embed)

@commands.command(name="fusion_search")
async def fusion_search(self, ctx, *, query: str):
    """
    RAG Fusion with multi-query synthesis.

    Usage: !fusion_search best python frameworks
    """
    await ctx.send(f"🔀 RAG Fusion: `{query}`")

    result = await self.rag_fusion_service.generate_fusion_answer(query)

    embed = discord.Embed(title="🔀 RAG Fusion Results", color=discord.Color.gold())

    # Generated queries
    queries_text = "\n".join(f"{i}. {q}" for i, q in enumerate(result["generated_queries"], 1))
    embed.add_field(
        name="📝 Generated Query Variations",
        value=queries_text,
        inline=False
    )

    # Answer
    embed.add_field(
        name="💡 Fused Answer",
        value=result["answer"],
        inline=False
    )

    # Sources
    sources_text = "\n".join(result["sources"][:5])
    embed.add_field(
        name="📚 Sources",
        value=sources_text,
        inline=False
    )

    await ctx.send(embed=embed)
```

---

## 7. Comparison & Evaluation

### When to Use Each Technique

| Technique | Best For | Pros | Cons |
|-----------|----------|------|------|
| **HyDE** | Question-answering | Bridges semantic gap | Extra LLM call |
| **Self-RAG** | High-precision needs | Quality control, transparency | Slower (3x LLM calls) |
| **RAG Fusion** | Complex/ambiguous queries | Comprehensive coverage | Higher cost (multiple searches) |
| **Iterative RAG** | Multi-hop reasoning | Handles complexity | Very slow |
| **Adaptive RAG** | Mixed workload | Best of all worlds | Complex to maintain |

### Performance Comparison

Create test suite:

```python
# Test query set
test_queries = [
    ("What is Python?", "simple"),
    ("Compare Django vs Flask for beginners", "complex"),
    ("Best practices for web development", "exploratory")
]

# Compare all techniques
for query, query_type in test_queries:
    results = {
        "basic": await basic_search(query),
        "hyde": await hyde_search(query),
        "self_rag": await self_rag(query),
        "fusion": await fusion_search(query)
    }

    # Evaluate each
    for method, result in results.items():
        precision = calculate_precision(result)
        latency = result["latency_ms"]
        cost = result["llm_tokens"] * COST_PER_TOKEN

        print(f"{method}: P={precision:.2f}, Latency={latency}ms, Cost=${cost:.4f}")
```

---

## Summary

### Advanced RAG Techniques Learned

1. **HyDE**: Generate hypothetical answers to bridge query-document semantic gap
2. **Self-RAG**: Three-stage reflection for quality control
3. **RAG Fusion**: Multi-query generation and result fusion
4. **Iterative RAG**: Multi-step retrieval for complex reasoning
5. **Adaptive RAG**: Dynamic strategy selection

### Key Takeaways

✅ **Different queries need different strategies**
✅ **More complex ≠ always better** (consider speed/cost trade-offs)
✅ **Combine techniques** (e.g., HyDE + RAG Fusion + Reranking)
✅ **Measure everything** (precision, recall, latency, cost)

### Next Steps

**Phase 17**: Build a comprehensive comparison dashboard to visualize and toggle between all RAG strategies we've learned!

---

## References

- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [RAG Fusion Blog Post](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)
- [Advanced RAG Techniques Survey](https://arxiv.org/abs/2312.10997)
