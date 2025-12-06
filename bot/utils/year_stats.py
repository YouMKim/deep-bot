"""
Statistics calculation utilities for year-in-review feature.
Calculates comprehensive user statistics from Discord message data.
"""

import re
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Filler words to exclude from word analysis
# Includes articles, pronouns, prepositions, common verbs, conjunctions, and adverbs
FILLER_WORDS = {
    # Articles
    'the', 'a', 'an',
    
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'myself', 'yourself', 'himself', 'hers', 'itself', 'ourselves', 'themselves',
    'this', 'that', 'these', 'those',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'onto', 'over', 'under', 'through', 'during', 'before', 'after',
    'between', 'among', 'within', 'without', 'about', 'above', 'below',
    'against', 'along', 'around', 'across', 'behind', 'beyond', 'inside',
    'outside', 'throughout', 'toward', 'towards', 'upon', 'via',
    
    # Conjunctions
    'and', 'or', 'but', 'if', 'when', 'where', 'while', 'because', 'since',
    'although', 'though', 'unless', 'until', 'whether', 'while',
    
    # Common auxiliary/helping verbs
    'be', 'been', 'being', 'am', 'is', 'are', 'was', 'were',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    'will', 'would', 'shall', 'should',
    'may', 'might', 'must', 'can', 'could',
    
    # Very common verbs
    'get', 'got', 'getting', 'gotten',
    'go', 'goes', 'went', 'gone', 'going',
    'come', 'comes', 'came', 'coming',
    'see', 'sees', 'saw', 'seen', 'seeing',
    'know', 'knows', 'knew', 'known', 'knowing',
    'think', 'thinks', 'thought', 'thinking',
    'say', 'says', 'said', 'saying',
    'tell', 'tells', 'told', 'telling',
    'make', 'makes', 'made', 'making',
    'take', 'takes', 'took', 'taken', 'taking',
    'give', 'gives', 'gave', 'given', 'giving',
    'want', 'wants', 'wanted', 'wanting',
    'need', 'needs', 'needed', 'needing',
    'like', 'likes', 'liked', 'liking',
    'try', 'tries', 'tried', 'trying',
    'use', 'uses', 'used', 'using',
    'find', 'finds', 'found', 'finding',
    'work', 'works', 'worked', 'working',
    'look', 'looks', 'looked', 'looking',
    'call', 'calls', 'called', 'calling',
    'ask', 'asks', 'asked', 'asking',
    'feel', 'feels', 'felt', 'feeling',
    'seem', 'seems', 'seemed', 'seeming',
    'leave', 'leaves', 'left', 'leaving',
    'let', 'lets', 'letting',
    'put', 'puts', 'putting',
    'show', 'shows', 'showed', 'shown', 'showing',
    
    # Common adverbs and intensifiers
    'very', 'really', 'quite', 'so', 'too', 'well',
    'just', 'only', 'also', 'even', 'still', 'already', 'yet',
    'more', 'most', 'less', 'least', 'much', 'many', 'most',
    'not', 'no', 'yes',
    'now', 'then', 'here', 'there', 'where',
    'how', 'why', 'what', 'who', 'when',
    'always', 'never', 'often', 'sometimes', 'usually',
    'all', 'some', 'any', 'every', 'each', 'both', 'either', 'neither',
    'once', 'twice', 'again', 'back',
    'up', 'down', 'out', 'off', 'away',
    
    # Common casual/interjection words
    'yeah', 'yep', 'yup', 'nah', 'nope',
    'ok', 'okay', 'alright', 'sure', 'right',
    'hey', 'hi', 'hello', 'bye', 'thanks', 'thank',
    'oh', 'ah', 'um', 'uh', 'hmm',
    'lol', 'haha', 'hahaha', 'lmao', 'rofl',
    'dunno', 'gonna', 'wanna', 'gotta',
    
    # Numbers as words (common in casual speech)
    'one', 'two', 'three', 'four', 'five',
    'first', 'second', 'third', 'last',
    
    # Common filler phrases (when split)
    'im', 'ive', 'id', 'ill', 'youre', 'youve', 'youd', 'youll',
    'hes', 'shes', 'its', 'were', 'theyre', 'theyve', 'theyd', 'theyll',
    'thats', 'theres', 'heres', 'wheres', 'whos', 'whats',
    'cant', 'wont', 'dont', 'doesnt', 'didnt', 'isnt', 'arent',
    'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wouldnt', 'couldnt', 'shouldnt',
}


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_chars(text: str) -> int:
    """Count characters in text."""
    return len(text)


def detect_emojis(text: str) -> List[str]:
    """
    Detect emojis in text using Unicode ranges.
    Returns list of emoji strings found.
    """
    # Unicode emoji patterns
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE
    )
    
    emojis = emoji_pattern.findall(text)
    return emojis


def calculate_basic_counts(messages: List[Dict]) -> Dict:
    """Calculate basic message counts."""
    total_messages = len(messages)
    total_words = sum(count_words(msg.get('content', '')) for msg in messages)
    total_chars = sum(count_chars(msg.get('content', '')) for msg in messages)
    
    avg_words = total_words / total_messages if total_messages > 0 else 0
    avg_chars = total_chars / total_messages if total_messages > 0 else 0
    
    return {
        'total_messages': total_messages,
        'total_words': total_words,
        'total_chars': total_chars,
        'avg_words': round(avg_words, 2),
        'avg_chars': round(avg_chars, 2)
    }


def calculate_message_extremes(messages: List[Dict], guild_id: Optional[str] = None) -> Dict:
    """Calculate longest and shortest messages."""
    if not messages:
        return {
            'longest': None,
            'shortest': None,
            'avg_length': 0
        }
    
    # Find longest and shortest by character count
    longest = max(messages, key=lambda m: len(m.get('content', '')))
    shortest = min(messages, key=lambda m: len(m.get('content', '')) if len(m.get('content', '')) > 0 else float('inf'))
    
    avg_length = sum(len(m.get('content', '')) for m in messages) / len(messages) if messages else 0
    
    # Build message links (same format as snapshot feature)
    longest_link = None
    longest_guild_id = longest.get('guild_id') or guild_id
    longest_channel_id = longest.get('channel_id')
    longest_message_id = longest.get('message_id')
    if longest_guild_id and longest_channel_id and longest_message_id:
        longest_link = f"https://discord.com/channels/{longest_guild_id}/{longest_channel_id}/{longest_message_id}"
    
    shortest_link = None
    shortest_guild_id = shortest.get('guild_id') or guild_id
    shortest_channel_id = shortest.get('channel_id')
    shortest_message_id = shortest.get('message_id')
    if shortest_guild_id and shortest_channel_id and shortest_message_id:
        shortest_link = f"https://discord.com/channels/{shortest_guild_id}/{shortest_channel_id}/{shortest_message_id}"
    
    return {
        'longest': {
            'content': longest.get('content', ''),
            'char_count': len(longest.get('content', '')),
            'message_id': longest.get('message_id'),
            'channel_id': longest.get('channel_id'),
            'link': longest_link
        },
        'shortest': {
            'content': shortest.get('content', ''),
            'char_count': len(shortest.get('content', '')),
            'message_id': shortest.get('message_id'),
            'channel_id': shortest.get('channel_id'),
            'link': shortest_link
        },
        'avg_length': round(avg_length, 2)
    }


def calculate_activity_patterns(messages: List[Dict]) -> Dict:
    """Calculate activity patterns from timestamps."""
    if not messages:
        return {
            'most_active_hour': None,
            'most_active_day': None,
            'most_active_month': None,
            'active_days': 0,
            'peak_day': None,
            'time_distribution': {},
            'longest_streak': 0
        }
    
    hour_counts = Counter()
    day_counts = Counter()  # 0=Monday, 6=Sunday
    month_counts = Counter()
    date_counts = Counter()  # For peak day
    unique_dates = set()
    
    for msg in messages:
        try:
            # Parse timestamp (Discord stores in UTC)
            timestamp_str = msg.get('timestamp', '').replace('Z', '+00:00')
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Ensure timestamp is timezone-aware (UTC)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if not already
                timestamp = timestamp.astimezone(timezone.utc)
            
            # Convert to Pacific timezone for activity analysis
            # Pacific timezone (handles both PST/PDT automatically)
            try:
                from zoneinfo import ZoneInfo
                pacific = ZoneInfo('America/Los_Angeles')
                pacific_timestamp = timestamp.astimezone(pacific)
            except ImportError:
                # Fallback for Python < 3.9: use UTC offset (-8 hours for PST, -7 for PDT)
                # Simple approximation: assume PST (UTC-8)
                from datetime import timedelta
                pacific_timestamp = timestamp - timedelta(hours=8)
            
            hour = pacific_timestamp.hour
            day_of_week = pacific_timestamp.weekday()
            month = pacific_timestamp.month
            date_key = pacific_timestamp.date()
            
            hour_counts[hour] += 1
            day_counts[day_of_week] += 1
            month_counts[month] += 1
            date_counts[date_key] += 1
            unique_dates.add(date_key)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse timestamp: {msg.get('timestamp')}, error: {e}")
            continue
    
    # Most active hour
    most_active_hour = hour_counts.most_common(1)[0][0] if hour_counts else None
    most_active_hour_count = hour_counts.most_common(1)[0][1] if hour_counts else 0
    
    # Most active day of week
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    most_active_day_num = day_counts.most_common(1)[0][0] if day_counts else None
    most_active_day = day_names[most_active_day_num] if most_active_day_num is not None else None
    most_active_day_count = day_counts.most_common(1)[0][1] if day_counts else 0
    
    # Most active month
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    most_active_month_num = month_counts.most_common(1)[0][0] if month_counts else None
    most_active_month = month_names[most_active_month_num] if most_active_month_num else None
    most_active_month_count = month_counts.most_common(1)[0][1] if month_counts else 0
    
    # Peak day
    peak_day = date_counts.most_common(1)[0][0] if date_counts else None
    peak_day_count = date_counts.most_common(1)[0][1] if date_counts else 0
    
    # Time distribution (morning/afternoon/evening/night)
    time_distribution = {
        'morning': 0,    # 6am-12pm
        'afternoon': 0,  # 12pm-6pm
        'evening': 0,    # 6pm-12am
        'night': 0       # 12am-6am
    }
    
    for hour, count in hour_counts.items():
        if 6 <= hour < 12:
            time_distribution['morning'] += count
        elif 12 <= hour < 18:
            time_distribution['afternoon'] += count
        elif 18 <= hour < 24:
            time_distribution['evening'] += count
        else:
            time_distribution['night'] += count
    
    # Get time preference (only if there are messages)
    time_preference = None
    if time_distribution and sum(time_distribution.values()) > 0:
        time_preference = max(time_distribution.items(), key=lambda x: x[1])[0]
    
    # Calculate longest streak
    sorted_dates = sorted(unique_dates)
    longest_streak = calculate_streak(sorted_dates)
    
    return {
        'most_active_hour': most_active_hour,
        'most_active_hour_count': most_active_hour_count,
        'most_active_day': most_active_day,
        'most_active_day_count': most_active_day_count,
        'most_active_month': most_active_month,
        'most_active_month_count': most_active_month_count,
        'active_days': len(unique_dates),
        'peak_day': {
            'date': peak_day,
            'count': peak_day_count
        } if peak_day else None,
        'time_distribution': time_distribution,
        'time_preference': time_preference,
        'longest_streak': longest_streak
    }


def calculate_streak(sorted_dates: List) -> int:
    """Calculate longest consecutive day streak."""
    if not sorted_dates:
        return 0
    
    from datetime import timedelta
    
    if len(sorted_dates) == 1:
        return 1
    
    longest = 1
    current = 1
    
    for i in range(1, len(sorted_dates)):
        diff = (sorted_dates[i] - sorted_dates[i-1]).days
        if diff == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    
    return longest


def calculate_channel_preferences(messages: List[Dict]) -> Dict:
    """Calculate channel preferences."""
    if not messages:
        return {
            'most_active_channel': None,
            'top_channels': [],
            'total_channels': 0
        }
    
    channel_counts = Counter()
    for msg in messages:
        channel_name = msg.get('channel_name', 'unknown')
        channel_counts[channel_name] += 1
    
    most_active_channel = channel_counts.most_common(1)[0][0] if channel_counts else None
    most_active_count = channel_counts.most_common(1)[0][1] if channel_counts else 0
    
    # Top 3 channels
    top_channels = [
        {'name': name, 'count': count}
        for name, count in channel_counts.most_common(3)
    ]
    
    return {
        'most_active_channel': {
            'name': most_active_channel,
            'count': most_active_count
        } if most_active_channel else None,
        'top_channels': top_channels,
        'total_channels': len(channel_counts)
    }


def calculate_emoji_stats(messages: List[Dict], guild_id: Optional[str] = None) -> Dict:
    """Calculate emoji usage statistics."""
    all_emojis = []
    messages_with_emojis = 0
    emoji_per_message = []
    
    for msg in messages:
        content = msg.get('content', '')
        emojis = detect_emojis(content)
        
        if emojis:
            messages_with_emojis += 1
            all_emojis.extend(emojis)
            emoji_per_message.append({
                'message': msg,
                'emoji_count': len(emojis),
                'emojis': emojis
            })
    
    total_emojis = len(all_emojis)
    emoji_usage_pct = (messages_with_emojis / len(messages) * 100) if messages else 0
    
    # Top 3 emojis
    emoji_counts = Counter(all_emojis)
    top_emojis = [
        {'emoji': emoji, 'count': count}
        for emoji, count in emoji_counts.most_common(3)
    ]
    
    # Message with most emojis (random tiebreaker)
    if emoji_per_message:
        max_emoji_count = max(e['emoji_count'] for e in emoji_per_message)
        messages_with_max = [e for e in emoji_per_message if e['emoji_count'] == max_emoji_count]
        most_emoji_msg = random.choice(messages_with_max)['message']
        
        # Build link (same format as snapshot feature)
        most_emoji_link = None
        emoji_guild_id = most_emoji_msg.get('guild_id') or guild_id
        emoji_channel_id = most_emoji_msg.get('channel_id')
        emoji_message_id = most_emoji_msg.get('message_id')
        if emoji_guild_id and emoji_channel_id and emoji_message_id:
            most_emoji_link = f"https://discord.com/channels/{emoji_guild_id}/{emoji_channel_id}/{emoji_message_id}"
        
        most_emoji = {
            'content': most_emoji_msg.get('content', ''),
            'emoji_count': max_emoji_count,
            'message_id': most_emoji_msg.get('message_id'),
            'channel_id': most_emoji_msg.get('channel_id'),
            'link': most_emoji_link
        }
    else:
        most_emoji = None
    
    return {
        'total_emojis': total_emojis,
        'top_emojis': top_emojis,
        'emoji_usage_pct': round(emoji_usage_pct, 1),
        'most_emoji_message': most_emoji
    }


def calculate_word_frequency(messages: List[Dict]) -> List[Dict]:
    """Calculate top words excluding filler words and common glue words."""
    all_words = []
    
    for msg in messages:
        content = msg.get('content', '').lower()
        # Extract words - handle contractions and split on punctuation
        # Remove URLs, mentions, and emoji-like patterns first
        content = re.sub(r'https?://\S+|@\w+|#\w+', '', content)
        words = re.findall(r'\b[a-z]+(?:[’\']?[a-z]+)?\b', content)
        
        # Filter out filler words and very short words (must be 3+ chars)
        words = [
            w for w in words 
            if len(w) >= 3 
            and w not in FILLER_WORDS 
            and not w.isdigit()  # Exclude pure numbers
            and not re.match(r'^[a-z]{1,2}$', w)  # Exclude 1-2 letter words
        ]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_words = [
        {'word': word, 'count': count}
        for word, count in word_counts.most_common(5)
    ]
    
    return top_words


def calculate_user_stats(
    messages: List[Dict],
    guild_id: Optional[str] = None
) -> Dict:
    """
    Calculate comprehensive statistics for a user from their messages.
    
    Args:
        messages: List of message dictionaries
        guild_id: Optional guild ID for generating message links
        
    Returns:
        Dictionary with all calculated statistics
    """
    if not messages:
        return {
            'basic_counts': calculate_basic_counts([]),
            'extremes': calculate_message_extremes([]),
            'activity': calculate_activity_patterns([]),
            'channels': calculate_channel_preferences([]),
            'emojis': calculate_emoji_stats([]),
            'words': []
        }
    
    stats = {
        'basic_counts': calculate_basic_counts(messages),
        'extremes': calculate_message_extremes(messages, guild_id),
        'activity': calculate_activity_patterns(messages),
        'channels': calculate_channel_preferences(messages),
        'emojis': calculate_emoji_stats(messages, guild_id),
        'words': calculate_word_frequency(messages)
    }
    
    return stats

