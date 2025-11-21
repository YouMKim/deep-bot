#!/usr/bin/env python3
"""
Script to wipe/clear all social credit data.

WARNING: This will delete ALL social credit scores and history!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.social_credit import SocialCreditManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wipe_social_credit():
    """Wipe all social credit data."""
    manager = SocialCreditManager()
    
    with manager._get_connection() as conn:
        cursor = conn.cursor()
        
        # Count records before deletion
        cursor.execute("SELECT COUNT(*) FROM user_ai_stats")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM social_credit_history")
        history_count = cursor.fetchone()[0]
        
        logger.info(f"Found {user_count} users and {history_count} history records")
        
        # Delete all records
        cursor.execute("DELETE FROM social_credit_history")
        cursor.execute("DELETE FROM user_ai_stats")
        
        conn.commit()
        
        logger.info(f"✅ Deleted {user_count} users and {history_count} history records")
        logger.info("✅ Social credit tables have been wiped!")

if __name__ == "__main__":
    import sys
    
    print("⚠️  WARNING: This will delete ALL social credit data!")
    
    # Check for --confirm flag
    if len(sys.argv) > 1 and sys.argv[1] == "--confirm":
        wipe_social_credit()
    else:
        print("❌ This script requires --confirm flag to run.")
        print("Usage: python scripts/wipe_social_credit.py --confirm")
        sys.exit(1)

