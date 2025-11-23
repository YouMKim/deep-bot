import logging
import os
from typing import List, Optional, TYPE_CHECKING
from chunking.base import Chunk
from chunking.constants import ChunkStrategy
from .embedding_service import EmbeddingService
from storage.vectors.base import VectorStorage

if TYPE_CHECKING:
    from config import Config


class BotKnowledgeService:
    """Service for ingesting bot documentation into ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStorage,
        embedding_service: EmbeddingService,
        config: Optional['Config'] = None
    ):
        """
        Initialize BotKnowledgeService.

        Args:
            vector_store: VectorStorage instance for storing chunks
            embedding_service: EmbeddingService instance for generating embeddings
            config: Configuration instance (defaults to Config class)
        """
        from config import Config as ConfigClass
        
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)
        
        # Get path to bot_knowledge.md
        # __file__ is storage/chunked_memory/bot_knowledge_service.py
        # We want docs/bot_knowledge.md relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.bot_knowledge_path = os.path.join(project_root, "docs", "bot_knowledge.md")

    def load_bot_documentation(self) -> str:
        """
        Load bot documentation from markdown file.

        Returns:
            Documentation content as string
        """
        try:
            if not os.path.exists(self.bot_knowledge_path):
                self.logger.warning(f"Bot knowledge file not found: {self.bot_knowledge_path}")
                return ""
            
            with open(self.bot_knowledge_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.info(f"Loaded bot documentation ({len(content)} characters)")
            return content
        except Exception as e:
            self.logger.error(f"Failed to load bot documentation: {e}", exc_info=True)
            return ""

    def chunk_bot_docs(self, content: str) -> List[Chunk]:
        """
        Split bot documentation into chunks.

        Args:
            content: Documentation content

        Returns:
            List of Chunk objects
        """
        if not content:
            return []

        chunks = []
        
        # Split by markdown headers (## or ###)
        import re
        # Split content by headers, keeping headers with content
        parts = re.split(r'\n(##+)\s+(.+?)(?=\n##|\Z)', content, flags=re.DOTALL)
        
        current_section = ""
        current_section_name = "overview"
        chunk_id = 0
        
        # Process parts
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue
            
            # Check if this is a header marker (##+)
            if part.startswith('#'):
                # Next part should be header text
                if i + 1 < len(parts):
                    header_text = parts[i + 1].strip()
                    # Save previous section
                    if current_section:
                        section_chunks = self._split_section_into_chunks(
                            current_section,
                            current_section_name,
                            chunk_id
                        )
                        chunks.extend(section_chunks)
                        chunk_id += len(section_chunks)
                    
                    # Start new section
                    current_section_name = header_text.lower().replace(' ', '_').replace('#', '')
                    current_section = ""
                    i += 2
                else:
                    i += 1
            else:
                # This is content
                if current_section:
                    current_section += "\n\n" + part
                else:
                    current_section = part
                i += 1
        
        # Add final section
        if current_section:
            section_chunks = self._split_section_into_chunks(
                current_section,
                current_section_name,
                chunk_id
            )
            chunks.extend(section_chunks)
        
        self.logger.info(f"Created {len(chunks)} chunks from bot documentation")
        return chunks

    def _split_section_into_chunks(
        self,
        content: str,
        section_name: str,
        start_id: int
    ) -> List[Chunk]:
        """
        Split a section into chunks if it's too long.

        Args:
            content: Section content
            section_name: Name of the section
            start_id: Starting chunk ID

        Returns:
            List of Chunk objects
        """
        chunks = []
        max_chunk_size = 1000  # characters per chunk
        
        if len(content) <= max_chunk_size:
            # Single chunk
            chunk = Chunk(
                content=content,
                message_ids=[f"bot_doc_{start_id}"],
                metadata={
                    'chunk_strategy': 'bot_docs',
                    'channel_id': 'system',
                    'author': 'system',
                    'source': 'bot_documentation',
                    'section': section_name,
                    'first_message_id': f"bot_doc_{start_id}",
                    'timestamp': '2024-01-01T00:00:00Z',
                    'author_display_name': 'Deep-Bot Documentation',
                }
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            import re
            # Split by paragraphs (double newlines)
            paragraphs = re.split(r'\n\n+', content)
            current_chunk = ""
            chunk_idx = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        message_ids=[f"bot_doc_{start_id + chunk_idx}"],
                        metadata={
                            'chunk_strategy': 'bot_docs',
                            'channel_id': 'system',
                            'author': 'system',
                            'source': 'bot_documentation',
                            'section': section_name,
                            'first_message_id': f"bot_doc_{start_id + chunk_idx}",
                            'timestamp': '2024-01-01T00:00:00Z',
                            'author_display_name': 'Deep-Bot Documentation',
                        }
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            # Add final chunk
            if current_chunk.strip():
                chunk = Chunk(
                    content=current_chunk.strip(),
                    message_ids=[f"bot_doc_{start_id + chunk_idx}"],
                    metadata={
                        'chunk_strategy': 'bot_docs',
                        'channel_id': 'system',
                        'author': 'system',
                        'source': 'bot_documentation',
                        'section': section_name,
                        'first_message_id': f"bot_doc_{start_id + chunk_idx}",
                        'timestamp': '2024-01-01T00:00:00Z',
                        'author_display_name': 'Deep-Bot Documentation',
                    }
                )
                chunks.append(chunk)
        
        return chunks

    async def ingest_bot_knowledge(self, force: bool = False) -> bool:
        """
        Ingest bot documentation into ChromaDB.

        Args:
            force: If True, re-index even if already indexed

        Returns:
            True if successful, False otherwise
        """
        try:
            from .utils import get_collection_name
            
            collection_name = get_collection_name('bot_docs')
            
            # Check if already indexed
            if not force:
                try:
                    collections = self.vector_store.list_collections()
                    if collection_name in collections:
                        # Try to get count by querying with a dummy embedding
                        # Get embedding dimension from embedder
                        dummy_text = "test"
                        dummy_embedding = self.embedding_service.embedder.encode(dummy_text)
                        
                        results = self.vector_store.query(
                            collection_name=collection_name,
                            query_embeddings=[dummy_embedding],
                            n_results=1
                        )
                        if results and results.get('ids') and len(results['ids'][0]) > 0:
                            self.logger.info("Bot knowledge already indexed, skipping")
                            return True
                except Exception as e:
                    self.logger.debug(f"Could not check existing index: {e}")
                    # Continue with indexing
            
            # Load documentation
            content = self.load_bot_documentation()
            if not content:
                self.logger.error("No bot documentation content to index")
                return False
            
            # Chunk documentation
            chunks = self.chunk_bot_docs(content)
            if not chunks:
                self.logger.error("No chunks created from bot documentation")
                return False
            
            # Create collection
            self.vector_store.create_collection(collection_name)
            
            # Prepare documents, embeddings, and metadata
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.metadata['first_message_id'] for chunk in chunks]
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(chunks)} bot knowledge chunks")
            embeddings = await self.embedding_service.embed_in_batches(documents)
            
            # Store in vector database
            self.vector_store.add_documents(
                collection_name=collection_name,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            
            self.logger.info(f"Successfully indexed {len(chunks)} bot knowledge chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ingest bot knowledge: {e}", exc_info=True)
            return False

