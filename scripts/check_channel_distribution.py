#!/usr/bin/env python3
"""
Script to check channel distribution in collections.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.vectors.factory import VectorStoreFactory
from collections import Counter
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def check_channel_distribution():
    """Check which channels are represented in collections."""
    vector_store = VectorStoreFactory.create()
    
    collections = vector_store.list_collections()
    discord_collections = [c for c in collections if c.startswith("discord_chunks_")]
    
    print("\n🔍 Checking channel distribution in collections:\n")
    
    for collection_name in sorted(discord_collections):
        count = vector_store.get_collection_count(collection_name)
        if count == 0:
            print(f"  {collection_name}: (empty)")
            continue
        
        # Get a sample of documents to check channel_id distribution
        try:
            collection = vector_store.get_collection(collection_name)
            # Get all documents (or a large sample)
            results = collection.get(limit=min(10000, count))  # Sample up to 10k
            
            channel_ids = []
            for metadata in results.get('metadatas', []):
                if metadata and 'channel_id' in metadata:
                    channel_ids.append(metadata['channel_id'])
            
            if channel_ids:
                channel_counter = Counter(channel_ids)
                unique_channels = len(channel_counter)
                print(f"  {collection_name}:")
                print(f"    Total documents: {count:,}")
                print(f"    Unique channels: {unique_channels}")
                if unique_channels <= 10:
                    print(f"    Channels: {', '.join(channel_counter.keys())}")
                else:
                    top_channels = channel_counter.most_common(5)
                    print(f"    Top 5 channels:")
                    for channel_id, doc_count in top_channels:
                        print(f"      - {channel_id}: {doc_count:,} documents")
            else:
                print(f"  {collection_name}: No channel_id found in metadata")
                
        except Exception as e:
            print(f"  {collection_name}: Error - {e}")

if __name__ == "__main__":
    check_channel_distribution()

