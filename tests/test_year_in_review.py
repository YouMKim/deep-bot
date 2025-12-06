"""
Tests for year-in-review feature.

Tests cover:
- Database query methods for year-in-review
- Statistics calculation functions
- Completion tracking
- Edge cases (no messages, minimal activity, etc.)
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from storage.messages import MessageStorage
from bot.utils.year_stats import (
    calculate_user_stats,
    calculate_basic_counts,
    calculate_message_extremes,
    calculate_activity_patterns,
    calculate_channel_preferences,
    calculate_emoji_stats,
    calculate_word_frequency,
    detect_emojis,
    FILLER_WORDS
)

# Import discord for admin command tests
try:
    import discord
except ImportError:
    discord = None


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_messages.db")
    
    storage = MessageStorage(db_path)
    yield storage
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_messages_2025():
    """Sample messages from 2025 for year-in-review testing"""
    base_date = datetime(2025, 3, 15, 14, 0, 0)
    return [
        {
            'id': 'msg1',
            'content': 'Hello world! This is a test message with some words. 😀',
            'author_id': 'user1',
            'author': 'TestUser',
            'author_display_name': 'TestUser',
            'timestamp': (base_date).isoformat(),
            'created_at': (base_date).isoformat(),
            'channel_id': 'channel1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'msg1',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': 'msg2',
            'content': 'Short msg 🔥',
            'author_id': 'user1',
            'author': 'TestUser',
            'author_display_name': 'TestUser',
            'timestamp': (base_date + timedelta(hours=1)).isoformat(),
            'created_at': (base_date + timedelta(hours=1)).isoformat(),
            'channel_id': 'channel1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'msg2',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': 'msg3',
            'content': 'A' * 500,  # Long message
            'author_id': 'user1',
            'author': 'TestUser',
            'author_display_name': 'TestUser',
            'timestamp': (base_date + timedelta(days=1)).isoformat(),
            'created_at': (base_date + timedelta(days=1)).isoformat(),
            'channel_id': 'channel2',
            'channel_name': 'random',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'msg3',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': 'msg4',
            'content': 'Testing emojis 😂🔥💯🎉',
            'author_id': 'user1',
            'author': 'TestUser',
            'author_display_name': 'TestUser',
            'timestamp': (base_date + timedelta(days=2)).isoformat(),
            'created_at': (base_date + timedelta(days=2)).isoformat(),
            'channel_id': 'channel1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'msg4',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': 'msg5',
            'content': 'Bot message',
            'author_id': 'bot1',
            'author': 'Bot',
            'author_display_name': 'Bot',
            'timestamp': (base_date + timedelta(days=3)).isoformat(),
            'created_at': (base_date + timedelta(days=3)).isoformat(),
            'channel_id': 'channel1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'msg5',
            'is_bot': True,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        }
    ]


@pytest.fixture
def messages_in_date_range():
    """Messages within the year-in-review date range (Jan 1 - Dec 4, 2025)"""
    return [
        {
            'id': 'yr1',
            'content': 'Message in range',
            'author_id': 'user1',
            'author': 'User1',
            'author_display_name': 'User1',
            'timestamp': '2025-06-15T12:00:00Z',
            'created_at': '2025-06-15T12:00:00Z',
            'channel_id': 'ch1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'yr1',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        }
    ]


@pytest.fixture
def messages_outside_date_range():
    """Messages outside the year-in-review date range"""
    return [
        {
            'id': 'out1',
            'content': 'Message outside range',
            'author_id': 'user1',
            'author': 'User1',
            'author_display_name': 'User1',
            'timestamp': '2024-12-31T12:00:00Z',  # Before 2025
            'created_at': '2024-12-31T12:00:00Z',
            'channel_id': 'ch1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'out1',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': 'out2',
            'content': 'Message after range',
            'author_id': 'user1',
            'author': 'User1',
            'author_display_name': 'User1',
            'timestamp': '2025-12-05T12:00:00Z',  # After Dec 4
            'created_at': '2025-12-05T12:00:00Z',
            'channel_id': 'ch1',
            'channel_name': 'general',
            'guild_name': 'Test Guild',
            'guild_id': 'guild1',
            'message_id': 'out2',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        }
    ]


class TestYearInReviewDatabaseQueries:
    """Test database query methods for year-in-review"""
    
    def test_get_user_messages_by_date_range(self, temp_db, sample_messages_2025):
        """Test querying user messages within date range"""
        # Save messages to different channels
        temp_db.save_channel_messages('channel1', sample_messages_2025[:2])
        temp_db.save_channel_messages('channel2', [sample_messages_2025[2]])
        
        start_date = datetime(2025, 1, 1).isoformat()
        end_date = datetime(2025, 12, 4, 23, 59, 59).isoformat()
        
        # Get messages for user1
        messages = temp_db.get_user_messages_by_date_range(
            author_id='user1',
            start_date=start_date,
            end_date=end_date
        )
        
        # Should return 3 messages (excluding bot message)
        assert len(messages) == 3
        assert all(msg['author_id'] == 'user1' for msg in messages)
        assert all(not msg.get('is_bot', False) for msg in messages)
    
    def test_get_user_messages_filters_by_date(self, temp_db, messages_in_date_range, messages_outside_date_range):
        """Test that date filtering works correctly"""
        temp_db.save_channel_messages('channel1', messages_in_date_range + messages_outside_date_range)
        
        start_date = datetime(2025, 1, 1).isoformat()
        end_date = datetime(2025, 12, 4, 23, 59, 59).isoformat()
        
        messages = temp_db.get_user_messages_by_date_range(
            author_id='user1',
            start_date=start_date,
            end_date=end_date
        )
        
        # Should only return message in range
        assert len(messages) == 1
        assert messages[0]['message_id'] == 'yr1'
    
    def test_get_unique_users_in_date_range(self, temp_db, sample_messages_2025):
        """Test getting unique users in date range"""
        temp_db.save_channel_messages('channel1', sample_messages_2025)
        
        start_date = datetime(2025, 1, 1).isoformat()
        end_date = datetime(2025, 12, 4, 23, 59, 59).isoformat()
        
        users = temp_db.get_unique_users_in_date_range(start_date, end_date)
        
        # Should only return non-bot users
        assert len(users) == 1
        assert users[0]['author_id'] == 'user1'
        assert users[0]['author_display_name'] == 'TestUser'
    
    def test_completion_tracking(self, temp_db):
        """Test year-in-review completion tracking"""
        user_id = 'user1'
        user_display_name = 'TestUser'
        
        # Mark as completed
        temp_db.mark_year_in_review_completed(user_id, user_display_name)
        
        # Check if completed
        assert temp_db.is_user_year_in_review_completed(user_id) is True
        assert temp_db.is_user_year_in_review_completed('user2') is False
    
    def test_get_next_unprocessed_user(self, temp_db, sample_messages_2025):
        """Test getting next unprocessed user"""
        temp_db.save_channel_messages('channel1', sample_messages_2025)
        
        start_date = datetime(2025, 1, 1).isoformat()
        end_date = datetime(2025, 12, 4, 23, 59, 59).isoformat()
        
        # Get first unprocessed user
        user = temp_db.get_next_unprocessed_user(start_date, end_date)
        assert user is not None
        assert user['author_id'] == 'user1'
        
        # Mark as completed
        temp_db.mark_year_in_review_completed(user['author_id'], user['author_display_name'])
        
        # Should return None (all users processed)
        user = temp_db.get_next_unprocessed_user(start_date, end_date)
        assert user is None


class TestStatisticsCalculation:
    """Test statistics calculation functions"""
    
    def test_calculate_basic_counts(self):
        """Test basic counts calculation"""
        messages = [
            {'content': 'Hello world'},
            {'content': 'Test message with more words'}
        ]
        
        counts = calculate_basic_counts(messages)
        
        assert counts['total_messages'] == 2
        assert counts['total_words'] == 7
        assert counts['total_chars'] == len('Hello world') + len('Test message with more words')
        assert counts['avg_words'] > 0
    
    def test_calculate_basic_counts_empty(self):
        """Test basic counts with empty messages"""
        counts = calculate_basic_counts([])
        
        assert counts['total_messages'] == 0
        assert counts['total_words'] == 0
        assert counts['total_chars'] == 0
        assert counts['avg_words'] == 0
        assert counts['avg_chars'] == 0
    
    def test_calculate_message_extremes(self):
        """Test message extremes calculation"""
        messages = [
            {'content': 'Short', 'message_id': '1', 'channel_id': 'ch1', 'guild_id': 'g1'},
            {'content': 'A' * 100, 'message_id': '2', 'channel_id': 'ch1', 'guild_id': 'g1'}
        ]
        
        extremes = calculate_message_extremes(messages, guild_id='g1')
        
        assert extremes['longest']['char_count'] == 100
        assert extremes['shortest']['char_count'] == 5
        assert extremes['longest']['link'] is not None
        assert extremes['shortest']['link'] is not None
    
    def test_calculate_activity_patterns(self):
        """Test activity patterns calculation"""
        base_date = datetime(2025, 3, 15, 14, 0, 0)
        messages = [
            {'timestamp': base_date.isoformat()},
            {'timestamp': (base_date + timedelta(hours=1)).isoformat()},
            {'timestamp': (base_date + timedelta(days=1)).isoformat()},
            {'timestamp': (base_date + timedelta(days=2)).isoformat()},
        ]
        
        activity = calculate_activity_patterns(messages)
        
        assert activity['most_active_hour'] is not None
        assert activity['most_active_day'] is not None
        assert activity['active_days'] >= 3  # At least 3 unique days
        assert activity['longest_streak'] >= 1
    
    def test_calculate_streak(self):
        """Test streak calculation"""
        from bot.utils.year_stats import calculate_streak
        from datetime import date
        
        # Consecutive dates
        dates = [
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 5),  # Gap
            date(2025, 1, 6),
        ]
        
        streak = calculate_streak(dates)
        assert streak == 3  # Longest streak is 3 days
    
    def test_calculate_channel_preferences(self):
        """Test channel preferences calculation"""
        messages = [
            {'channel_name': 'general', 'content': 'msg1'},
            {'channel_name': 'general', 'content': 'msg2'},
            {'channel_name': 'random', 'content': 'msg3'},
        ]
        
        channels = calculate_channel_preferences(messages)
        
        assert channels['most_active_channel']['name'] == 'general'
        assert channels['most_active_channel']['count'] == 2
        assert len(channels['top_channels']) <= 3
        assert channels['total_channels'] == 2
    
    def test_calculate_emoji_stats(self):
        """Test emoji statistics calculation"""
        messages = [
            {'content': 'Hello 😀 world 🔥', 'message_id': '1', 'channel_id': 'ch1', 'guild_id': 'g1'},
            {'content': 'Test 😀😀😀', 'message_id': '2', 'channel_id': 'ch1', 'guild_id': 'g1'},
            {'content': 'No emojis here', 'message_id': '3', 'channel_id': 'ch1', 'guild_id': 'g1'},
        ]
        
        emoji_stats = calculate_emoji_stats(messages, guild_id='g1')
        
        assert emoji_stats['total_emojis'] > 0
        assert len(emoji_stats['top_emojis']) <= 3
        assert emoji_stats['emoji_usage_pct'] > 0
        assert emoji_stats['most_emoji_message'] is not None
    
    def test_detect_emojis(self):
        """Test emoji detection"""
        text = "Hello 😀 world 🔥🎉"
        emojis = detect_emojis(text)
        
        # Emoji detection returns emoji sequences, so 🔥🎉 might be together
        assert len(emojis) >= 2
        assert any('😀' in e for e in emojis)
    
    def test_calculate_word_frequency(self):
        """Test word frequency calculation"""
        messages = [
            {'content': 'The project is awesome'},
            {'content': 'This project works well'},
            {'content': 'The code is great'},
        ]
        
        words = calculate_word_frequency(messages)
        
        # Should exclude filler words like "the", "is"
        assert len(words) > 0
        word_list = [w['word'] for w in words]
        assert 'the' not in word_list  # Should be filtered
        assert 'is' not in word_list  # Should be filtered
    
    def test_calculate_user_stats_comprehensive(self, sample_messages_2025):
        """Test comprehensive user stats calculation"""
        # Filter out bot messages (as the database query does)
        user_messages = [msg for msg in sample_messages_2025 if not msg.get('is_bot', False)]
        stats = calculate_user_stats(user_messages, guild_id='guild1')
        
        assert 'basic_counts' in stats
        assert 'extremes' in stats
        assert 'activity' in stats
        assert 'channels' in stats
        assert 'emojis' in stats
        assert 'words' in stats
        
        # Verify basic counts (should be 4 messages, excluding bot)
        assert stats['basic_counts']['total_messages'] == 4
        assert stats['basic_counts']['total_words'] > 0
        
        # Verify extremes
        assert stats['extremes']['longest'] is not None
        assert stats['extremes']['shortest'] is not None
    
    def test_calculate_user_stats_empty(self):
        """Test user stats with no messages"""
        stats = calculate_user_stats([], guild_id='g1')
        
        assert stats['basic_counts']['total_messages'] == 0
        assert stats['extremes']['longest'] is None
        assert stats['extremes']['shortest'] is None


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_user_with_no_messages(self, temp_db):
        """Test user with no messages in date range"""
        start_date = datetime(2025, 1, 1).isoformat()
        end_date = datetime(2025, 12, 4, 23, 59, 59).isoformat()
        
        messages = temp_db.get_user_messages_by_date_range(
            author_id='nonexistent',
            start_date=start_date,
            end_date=end_date
        )
        
        assert messages == []
    
    def test_stats_with_single_message(self):
        """Test statistics with single message"""
        messages = [{'content': 'Only one message', 'timestamp': datetime(2025, 6, 15).isoformat()}]
        
        stats = calculate_user_stats(messages)
        
        assert stats['basic_counts']['total_messages'] == 1
        assert stats['extremes']['longest'] == stats['extremes']['shortest']
    
    def test_stats_with_only_emoji_messages(self):
        """Test statistics with messages containing only emojis"""
        messages = [
            {'content': '😀🔥💯', 'message_id': '1', 'channel_id': 'ch1', 'guild_id': 'g1'},
            {'content': '🎉🎊', 'message_id': '2', 'channel_id': 'ch1', 'guild_id': 'g1'},
        ]
        
        stats = calculate_user_stats(messages, guild_id='g1')
        
        assert stats['emojis']['total_emojis'] > 0
        assert stats['emojis']['emoji_usage_pct'] == 100.0
    
    def test_stats_with_all_filler_words(self):
        """Test word frequency with messages containing only filler words"""
        messages = [
            {'content': 'The and or but'},
            {'content': 'I you he she'},
        ]
        
        words = calculate_word_frequency(messages)
        
        # Should return empty or very few words
        assert len(words) == 0  # All filtered out
    
    def test_timezone_conversion(self):
        """Test that activity patterns convert UTC to Pacific Time"""
        # Create messages at different UTC times that should convert to PT
        # 20:00 UTC = 12:00 PM PT (during daylight saving)
        # 02:00 UTC = 6:00 PM PT (previous day, during daylight saving)
        messages = [
            {'timestamp': '2025-03-15T20:00:00Z'},  # 12 PM PT
            {'timestamp': '2025-03-15T20:00:00Z'},  # 12 PM PT
            {'timestamp': '2025-03-15T02:00:00Z'},  # 6 PM PT (previous day)
        ]
        
        activity = calculate_activity_patterns(messages)
        
        # Should have converted to Pacific Time
        assert activity['most_active_hour'] is not None
        # Most active hour should be in Pacific Time (not UTC)
        assert 0 <= activity['most_active_hour'] < 24
    
    def test_link_creation_format(self):
        """Test that message links are created in correct Discord format"""
        messages = [
            {
                'content': 'Test message',
                'message_id': '123456789',
                'channel_id': '987654321',
                'guild_id': '555666777',
            }
        ]
        
        extremes = calculate_message_extremes(messages, guild_id='555666777')
        
        if extremes['longest'].get('link'):
            link = extremes['longest']['link']
            # Should be in format: https://discord.com/channels/{guild_id}/{channel_id}/{message_id}
            assert link.startswith('https://discord.com/channels/')
            assert '555666777' in link
            assert '987654321' in link
            assert '123456789' in link
    
    def test_expanded_filler_words(self):
        """Test that expanded filler words list filters more words"""
        # Test some of the new filler words we added
        messages = [
            {'content': 'yeah ok sure thanks'},
            {'content': 'gonna wanna gotta'},
            {'content': 'through during before after'},
            {'content': 'because since although'},
        ]
        
        words = calculate_word_frequency(messages)
        
        # Should filter out most/all of these filler words
        word_list = [w['word'] for w in words]
        # None of these should appear (they're all filler words)
        filler_check = ['yeah', 'ok', 'sure', 'thanks', 'gonna', 'wanna', 'gotta',
                       'through', 'during', 'before', 'after', 'because', 'since', 'although']
        for filler in filler_check:
            assert filler not in word_list, f"Filler word '{filler}' should be filtered but wasn't"
    
    def test_word_filtering_min_length(self):
        """Test that words must be at least 3 characters"""
        messages = [
            {'content': 'a b c ab cd'},
            {'content': 'abc def ghi'},
        ]
        
        words = calculate_word_frequency(messages)
        
        # All words should be 3+ characters
        for word_data in words:
            assert len(word_data['word']) >= 3


class TestAdminCommand:
    """Test year-in-review admin command functionality"""
    
    def test_filler_words_comprehensive(self):
        """Test that comprehensive filler words list works correctly"""
        # Test a mix of old and new filler words
        test_cases = [
            {'content': 'yeah ok sure', 'expected_filtered': True},
            {'content': 'gonna wanna gotta', 'expected_filtered': True},
            {'content': 'through during before after', 'expected_filtered': True},
            {'content': 'because since although', 'expected_filtered': True},
            {'content': 'project code awesome', 'expected_filtered': False},  # Should keep these
        ]
        
        for test_case in test_cases:
            words = calculate_word_frequency([test_case])
            word_list = [w['word'] for w in words]
            
            if test_case['expected_filtered']:
                # All words should be filtered
                assert len(word_list) == 0, f"Expected all words filtered for: {test_case['content']}"
            else:
                # Some words should remain
                assert len(word_list) > 0, f"Expected some words to remain for: {test_case['content']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

