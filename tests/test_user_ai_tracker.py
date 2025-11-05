"""
Tests for UserAITracker service.

Tests cover:
- AI usage logging
- Credit calculation
- User statistics retrieval
- Top users ranking
- Global statistics
"""

import pytest
import os
import tempfile
import shutil
from services.user_ai_tracker import UserAITracker


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_ai_usage.db")
    
    tracker = UserAITracker(db_path)
    yield tracker
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestUserAITracker:
    """Test suite for UserAITracker"""
    
    def test_log_ai_usage_new_user(self, temp_db):
        """Test logging AI usage for a new user"""
        temp_db.log_ai_usage(
            user_display_name="TestUser",
            cost=0.01,
            tokens_total=500
        )
        
        stats = temp_db.get_user_stats("TestUser")
        assert stats is not None
        assert stats['lifetime_cost'] == 0.01
        assert stats['lifetime_tokens'] == 500
        assert stats['lifetime_credit'] > 0  # Should have earned credit
    
    def test_log_ai_usage_existing_user(self, temp_db):
        """Test logging AI usage for existing user (accumulation)"""
        user_name = "TestUser"
        
        # First usage
        temp_db.log_ai_usage(user_name, cost=0.01, tokens_total=500)
        
        # Second usage
        temp_db.log_ai_usage(user_name, cost=0.02, tokens_total=300)
        
        stats = temp_db.get_user_stats(user_name)
        assert stats['lifetime_cost'] == 0.03  # 0.01 + 0.02
        assert stats['lifetime_tokens'] == 800  # 500 + 300
        assert stats['lifetime_credit'] > 0  # Should accumulate
    
    def test_credit_calculation(self, temp_db):
        """Test that credit is calculated correctly"""
        user_name = "TestUser"
        
        # Low cost, low tokens should give high credit
        temp_db.log_ai_usage(user_name, cost=0.0005, tokens_total=100)
        stats = temp_db.get_user_stats(user_name)
        
        # Base credit is 100, plus bonuses
        assert stats['lifetime_credit'] >= 100
    
    def test_get_user_stats_nonexistent(self, temp_db):
        """Test getting stats for non-existent user"""
        stats = temp_db.get_user_stats("NonexistentUser")
        assert stats is None
    
    def test_get_top_users(self, temp_db):
        """Test getting top users by credit"""
        # Create multiple users with different credit amounts
        users = [
            ("User1", 0.001, 50),   # Should get high credit
            ("User2", 0.01, 500),   # Medium credit
            ("User3", 0.1, 2000),  # Low credit
        ]
        
        for user, cost, tokens in users:
            temp_db.log_ai_usage(user, cost=cost, tokens_total=tokens)
        
        top_users = temp_db.get_top_users(limit=3)
        
        assert len(top_users) == 3
        # User1 should be first (lowest cost, lowest tokens = highest credit)
        assert top_users[0]['user_display_name'] == "User1"
        assert top_users[0]['lifetime_credit'] >= top_users[1]['lifetime_credit']
        assert top_users[1]['lifetime_credit'] >= top_users[2]['lifetime_credit']
    
    def test_get_top_users_limit(self, temp_db):
        """Test limiting the number of top users returned"""
        # Create 5 users
        for i in range(5):
            temp_db.log_ai_usage(f"User{i}", cost=0.01 * i, tokens_total=100 * i)
        
        # Request only top 2
        top_users = temp_db.get_top_users(limit=2)
        assert len(top_users) == 2
    
    def test_get_global_stats(self, temp_db):
        """Test getting global statistics"""
        # Log usage for multiple users
        temp_db.log_ai_usage("User1", cost=0.01, tokens_total=100)
        temp_db.log_ai_usage("User2", cost=0.02, tokens_total=200)
        temp_db.log_ai_usage("User3", cost=0.03, tokens_total=300)
        
        global_stats = temp_db.get_global_stats()
        
        assert global_stats['unique_users'] == 3
        assert global_stats['total_cost'] == 0.06  # 0.01 + 0.02 + 0.03
        assert global_stats['total_tokens'] == 600  # 100 + 200 + 300
    
    def test_get_global_stats_empty(self, temp_db):
        """Test global stats when no users exist"""
        global_stats = temp_db.get_global_stats()
        
        assert global_stats['unique_users'] == 0
        assert global_stats['total_cost'] == 0
        assert global_stats['total_tokens'] == 0
    
    def test_credit_accumulation(self, temp_db):
        """Test that credit accumulates across multiple uses"""
        user_name = "TestUser"
        
        # Multiple uses
        for i in range(5):
            temp_db.log_ai_usage(user_name, cost=0.01, tokens_total=100)
        
        stats = temp_db.get_user_stats(user_name)
        
        # Credit should be accumulated (5 uses * ~100 base credit each)
        assert stats['lifetime_credit'] >= 400  # At least 5 * base credit
        assert stats['lifetime_cost'] == 0.05  # 5 * 0.01
        assert stats['lifetime_tokens'] == 500  # 5 * 100
    
    def test_zero_cost_and_tokens(self, temp_db):
        """Test logging with zero cost and tokens"""
        user_name = "TestUser"
        temp_db.log_ai_usage(user_name, cost=0.0, tokens_total=0)
        
        stats = temp_db.get_user_stats(user_name)
        assert stats['lifetime_cost'] == 0.0
        assert stats['lifetime_tokens'] == 0
        assert stats['lifetime_credit'] > 0  # Should still get base credit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

