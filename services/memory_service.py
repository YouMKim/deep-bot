from data.chroma_client import chroma_client
from config import Config
from sentence_transformers import SentenceTransformer
import logging
from typing import Optional, List, Dict


class MemoryService:
    def __init__(self, db_path="data/chroma_db"):
        self.chroma_client = chroma_client
        self.collection = self.chroma_client.get_collection(
            name="discord_messages"
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.logger = logging.getLogger(__name__)

    def _create_message_metadata(self, message_data: dict) -> dict:
        return {
            # Channel information
            "channel_id": str(message_data.get("channel_id", "")),
            "channel_name": message_data.get("channel_name", "unknown"),
            "guild_id": str(message_data.get("guild_id", "")),
            "guild_name": message_data.get("guild_name", "unknown"),
            # Author information
            "author_id": str(message_data.get("author_id", "")),
            "author_name": message_data.get("author", "Unknown"),
            "author_display_name": message_data.get("author_display_name", "Unknown"),
            # Message information
            "message_id": str(message_data.get("id", "")),
            "timestamp": message_data.get("timestamp", ""),
            "created_at": message_data.get("created_at", ""),
            "is_bot": message_data.get("is_bot", False),
            # Content metadata
            "content_length": len(message_data.get("content", "")),
            "has_attachments": message_data.get("has_attachments", False),
            "message_type": message_data.get("message_type", "default"),
        }

    async def store_message(self, message_data: dict):
        try:
            content = message_data.get("content", "").strip()
            if not content or message_data.get("is_bot", False):
                return False

            metadata = self._create_message_metadata(message_data)
            embedding = self.embedder.encode(content)

            self.collection.add(
                documents=[content],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],  # Note: 'metadatas' not 'metadata'
                ids=[str(message_data.get("id", ""))],
            )
            return True
        except Exception as e:
            self.logger.error(f"Error storing message: {e}")
            return False

    async def find_relevant_messages(
        self,
        query: str,
        limit: int = 5,
        channel_id: Optional[str] = None,
        author_id: Optional[str] = None,
        guild_id: Optional[str] = None,
    ) -> List[Dict]:
        """Find relevant messages based on query and optional filters."""
        try:
            query_embedding = self.embedder.encode(query)

            where_clause = {}
            if channel_id:
                where_clause["channel_id"] = channel_id
            if author_id:
                where_clause["author_id"] = author_id
            if guild_id:
                where_clause["guild_id"] = guild_id

            # Search for relevant messages
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where=where_clause if where_clause else None,
            )

            # Format results with all metadata
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    formatted_results.append({
                        'content': doc,
                        'author_name': meta.get('author_name', 'Unknown'),
                        'author_display_name': meta.get('author_display_name', 'Unknown'),
                        'channel_name': meta.get('channel_name', 'unknown'),
                        'guild_name': meta.get('guild_name', 'unknown'),
                        'timestamp': meta.get('timestamp', ''),
                        'created_at': meta.get('created_at', ''),
                        'similarity_score': 1 - results['distances'][0][i],
                        'message_id': meta.get('message_id', ''),
                        'content_length': meta.get('content_length', 0)
                    })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error finding relevant messages: {e}")
            return []
    
    async def get_channel_stats(self, channel_id: str) -> Dict:
        """Get statistics for a specific channel"""
        try:
            # Count messages in specific channel
            channel_results = self.collection.query(
                query_texts=[""],  
                n_results=10000,  
                where={'channel_id': str(channel_id)}
            )
            
            channel_count = len(channel_results['documents'][0]) if channel_results['documents'] else 0
            
            return {
                'channel_id': channel_id,
                'message_count': channel_count,
                'total_messages': self.collection.count()
            }
        except Exception as e:
            self.logger.error(f"Error getting channel stats: {e}")
            return {'channel_id': channel_id, 'message_count': 0, 'total_messages': 0}
    
    def get_collection_stats(self) -> Dict:
        """Get overall collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_messages': count,
                'recent_messages': min(count, 10),
                'collection_name': 'discord_messages'
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'total_messages': 0, 'recent_messages': 0, 'collection_name': 'unknown'}
