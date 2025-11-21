#!/usr/bin/env python3
"""
Script to list all collections in the vector store and their document counts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.vectors.factory import VectorStoreFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_collections():
    """List all collections and their document counts."""
    vector_store = VectorStoreFactory.create()
    
    collections = vector_store.list_collections()
    
    if not collections:
        print("❌ No collections found.")
        return
    
    print(f"\n📚 Found {len(collections)} collection(s):\n")
    print("-" * 80)
    
    for collection_name in sorted(collections):
        count = vector_store.get_collection_count(collection_name)
        print(f"  • {collection_name:50s} ({count:,} documents)")
    
    print("-" * 80)
    print(f"\nTotal collections: {len(collections)}")
    
    # Show which are Discord-related
    discord_collections = [c for c in collections if c.startswith("discord_chunks_")]
    if discord_collections:
        print(f"\nDiscord collections: {len(discord_collections)}")
        for col in sorted(discord_collections):
            print(f"  - {col}")

if __name__ == "__main__":
    list_collections()

