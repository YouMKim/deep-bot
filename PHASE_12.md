# Phase 12: User Emulation Mode

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

User Emulation Mode - Mimic User Speech Patterns & Personality

### Learning Objectives
- Understand style transfer in NLP
- Learn personality modeling from text
- Practice user-specific RAG filtering
- Design ethical emulation features
- Implement speech pattern analysis

### Design Principles
- **Respect & Ethics**: Only emulate with permission, no impersonation
- **Accuracy**: Capture actual speech patterns, not stereotypes
- **Context-Aware**: Emulate how user would respond in THIS context
- **Transparency**: Always indicate it's emulation mode

### The Emulation Pipeline

```
User: "!emulate Alice what do you think about the new API design?"
    ↓
1. Identify target user (Alice)
2. Retrieve Alice's messages from RAG (user-filtered)
3. Analyze Alice's speech patterns:
   - Common phrases
   - Vocabulary
   - Tone (formal/casual)
   - Message length
   - Emoji usage
4. Find Alice's messages about similar topics
5. Generate response in Alice's style
    ↓
Bot: "🎭 [Emulating Alice]
     hmm the new API design looks pretty solid imo! i really like
     the REST endpoints but we might wanna add better error handling?
     just thinking about edge cases we ran into last time 🤔"
```

---

## Implementation Steps

### Step 12.1: Create User Style Analyzer

Create `services/user_style_analyzer.py`:

```python
"""
Analyze user's writing style and speech patterns.

Learning: Everyone has unique communication patterns:
- Word choice
- Sentence structure
- Punctuation
- Emoji usage
- Formality level
"""

from typing import List, Dict
import re
from collections import Counter
import logging

class UserStyleAnalyzer:
    """
    Analyze and model user's communication style.

    Learning: This is computational sociolinguistics!
    We're modeling individual speech patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_user_style(self, messages: List[Dict]) -> Dict:
        """
        Analyze user's communication style from their messages.

        Args:
            messages: List of messages from the user

        Returns:
            {
                "avg_message_length": 45,
                "vocabulary": ["cool", "awesome", "imo", ...],
                "common_phrases": ["i think", "imo", "ngl"],
                "emoji_usage": 0.3,  # 30% of messages have emoji
                "punctuation_style": "casual",  # or "formal"
                "capitalization_style": "mixed",  # or "proper", "lowercase"
                "formality_score": 0.4,  # 0=very casual, 1=very formal
                "greeting_style": ["hey", "hi guys"],
                "signature_phrases": ["just my 2 cents", "imo"]
            }
        """
        if not messages:
            return self._default_style()

        texts = [msg.get("content", "") for msg in messages if msg.get("content")]

        if not texts:
            return self._default_style()

        return {
            "avg_message_length": self._avg_length(texts),
            "vocabulary": self._extract_vocabulary(texts),
            "common_phrases": self._extract_common_phrases(texts),
            "emoji_usage": self._emoji_usage(texts),
            "punctuation_style": self._punctuation_style(texts),
            "capitalization_style": self._capitalization_style(texts),
            "formality_score": self._formality_score(texts),
            "greeting_style": self._extract_greetings(texts),
            "signature_phrases": self._extract_signature_phrases(texts)
        }

    def _avg_length(self, texts: List[str]) -> int:
        """Average message length in characters"""
        if not texts:
            return 0
        return sum(len(text) for text in texts) // len(texts)

    def _extract_vocabulary(self, texts: List[str], top_n: int = 50) -> List[str]:
        """
        Extract user's most common words.

        Learning: Everyone has favorite words they use frequently.
        This captures their unique vocabulary.
        """
        all_words = []
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)

        # Filter out very common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'or', 'but'}
        filtered_words = [w for w in all_words if w not in stop_words]

        # Get most common
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(top_n)]

    def _extract_common_phrases(self, texts: List[str], top_n: int = 20) -> List[str]:
        """
        Extract common 2-3 word phrases.

        Examples: "i think", "imo", "tbh", "ngl"
        """
        phrases = []
        for text in texts:
            text_lower = text.lower()
            # Extract 2-word phrases
            words = text_lower.split()
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase)

        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(top_n)]

    def _emoji_usage(self, texts: List[str]) -> float:
        """
        Calculate percentage of messages with emoji.

        Learning: Some people emoji every message, others never do.
        """
        # Simple emoji detection (Unicode emoji ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE
        )

        messages_with_emoji = sum(1 for text in texts if emoji_pattern.search(text))
        return messages_with_emoji / len(texts) if texts else 0

    def _punctuation_style(self, texts: List[str]) -> str:
        """
        Determine punctuation style.

        Casual: lots of "!", "...", no periods
        Formal: proper punctuation, periods at end
        """
        exclamation_count = sum(text.count('!') for text in texts)
        ellipsis_count = sum(text.count('...') for text in texts)
        period_count = sum(text.count('.') for text in texts)

        # Casual if lots of ! or ..., few periods
        if (exclamation_count + ellipsis_count) > period_count * 2:
            return "casual"
        else:
            return "formal"

    def _capitalization_style(self, texts: List[str]) -> str:
        """
        Determine capitalization style.

        - proper: Sentences start with capital
        - lowercase: everything lowercase
        - mixed: inconsistent
        """
        proper_count = 0
        lowercase_count = 0

        for text in texts:
            sentences = text.split('. ')
            for sentence in sentences:
                if sentence and len(sentence) > 1:
                    if sentence[0].isupper():
                        proper_count += 1
                    elif sentence[0].islower():
                        lowercase_count += 1

        if proper_count > lowercase_count * 2:
            return "proper"
        elif lowercase_count > proper_count * 2:
            return "lowercase"
        else:
            return "mixed"

    def _formality_score(self, texts: List[str]) -> float:
        """
        Score formality from 0 (very casual) to 1 (very formal).

        Indicators of formality:
        - Proper punctuation
        - Full words (not "u", "ur", "gonna")
        - Longer sentences
        - No slang
        """
        if not texts:
            return 0.5

        formality_indicators = 0
        total_checks = 0

        for text in texts:
            # Check 1: Ends with proper punctuation
            if text.strip() and text.strip()[-1] in '.!?':
                formality_indicators += 1
            total_checks += 1

            # Check 2: Uses full words (no "u", "ur")
            casual_words = ['u', 'ur', 'gonna', 'wanna', 'gotta', 'lol', 'idk']
            if not any(word in text.lower().split() for word in casual_words):
                formality_indicators += 1
            total_checks += 1

            # Check 3: Sentence length (formal = longer)
            if len(text.split()) > 10:
                formality_indicators += 1
            total_checks += 1

        return formality_indicators / total_checks if total_checks > 0 else 0.5

    def _extract_greetings(self, texts: List[str]) -> List[str]:
        """Extract how user greets people"""
        greeting_patterns = [
            r'\b(hey|hi|hello|sup|yo|heya|morning|afternoon)\b',
        ]

        greetings = []
        for text in texts:
            for pattern in greeting_patterns:
                matches = re.findall(pattern, text.lower())
                greetings.extend(matches)

        greeting_counts = Counter(greetings)
        return [g for g, c in greeting_counts.most_common(5)]

    def _extract_signature_phrases(self, texts: List[str]) -> List[str]:
        """
        Extract signature phrases user says often.

        Examples: "just my 2 cents", "imo", "ngl", "tbh"
        """
        signature_candidates = [
            "imo", "imho", "tbh", "ngl", "just saying",
            "my 2 cents", "just my opinion", "i think",
            "personally", "fwiw"
        ]

        found = []
        for text in texts:
            text_lower = text.lower()
            for phrase in signature_candidates:
                if phrase in text_lower:
                    found.append(phrase)

        phrase_counts = Counter(found)
        return [p for p, c in phrase_counts.most_common(5) if c >= 2]

    def _default_style(self) -> Dict:
        """Default style when no messages available"""
        return {
            "avg_message_length": 50,
            "vocabulary": [],
            "common_phrases": [],
            "emoji_usage": 0.0,
            "punctuation_style": "casual",
            "capitalization_style": "mixed",
            "formality_score": 0.5,
            "greeting_style": [],
            "signature_phrases": []
        }
```

### Step 12.2: Create User Emulation Service

Create `services/user_emulation_service.py`:

```python
"""
Emulate user's speech patterns and personality.

Learning: This combines:
1. Style analysis (how they speak)
2. RAG (what they know/think)
3. LLM (generate in their style)
"""

from ai.user_style_analyzer import UserStyleAnalyzer
from storage.chunked_memory import ChunkedMemoryService
from storage.messages import MessageStorage
from ai.service import AIService
from typing import Dict, Optional
from config import Config
import logging

class UserEmulationService:
    """
    Emulate how a specific user would respond.

    Learning: This is advanced RAG + style transfer!
    We combine content (what they'd say) with style (how they'd say it).
    """

    def __init__(
        self,
        message_storage: MessageStorage,
        chunked_memory: ChunkedMemoryService,
        ai_service: AIService
    ):
        self.storage = message_storage
        self.memory = chunked_memory
        self.ai = ai_service
        self.style_analyzer = UserStyleAnalyzer()
        self.logger = logging.getLogger(__name__)

    async def emulate_response(
        self,
        target_user_id: str,
        target_user_name: str,
        context: str,
        channel_id: str = None
    ) -> Dict[str, any]:
        """
        Generate response as if target user said it.

        Args:
            target_user_id: Discord user ID to emulate
            target_user_name: User's display name
            context: What to respond to (e.g., "what do you think about X?")
            channel_id: Optional channel filter

        Returns:
            {
                "response": "Emulated response in user's style",
                "confidence": 0.8,  # How confident we are
                "style_profile": {...},
                "sources": [...] # Messages used
            }
        """
        self.logger.info(f"Emulating user: {target_user_name} ({target_user_id})")

        # Step 1: Get user's messages
        all_messages = self.storage.load_channel_messages(channel_id) if channel_id else []
        user_messages = [
            msg for msg in all_messages
            if msg.get("author_id") == target_user_id
        ]

        if len(user_messages) < 10:
            return {
                "response": f"I don't have enough messages from {target_user_name} to emulate them accurately (found {len(user_messages)}, need at least 10).",
                "confidence": 0.0,
                "style_profile": None,
                "sources": []
            }

        # Step 2: Analyze user's style
        style_profile = self.style_analyzer.analyze_user_style(user_messages)

        # Step 3: Find relevant messages from user about similar topics
        # Use RAG but filter to only this user's messages
        relevant_messages = await self._find_relevant_user_messages(
            user_id=target_user_id,
            context=context,
            top_k=5
        )

        # Step 4: Build prompt for emulation
        prompt = self._build_emulation_prompt(
            target_user_name=target_user_name,
            context=context,
            style_profile=style_profile,
            example_messages=relevant_messages
        )

        # Step 5: Generate in user's style
        response = await self.ai.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.8  # Higher for more personality variation
        )

        confidence = self._calculate_confidence(user_messages, relevant_messages)

        return {
            "response": response.strip(),
            "confidence": confidence,
            "style_profile": style_profile,
            "sources": relevant_messages[:3]
        }

    async def _find_relevant_user_messages(
        self,
        user_id: str,
        context: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find user's messages relevant to context.

        Learning: User-filtered RAG!
        We only retrieve messages FROM this specific user.
        """
        # Get all chunks from memory
        all_chunks = self.memory.search(
            query=context,
            top_k=top_k * 3  # Get more, then filter
        )

        # Filter to only chunks containing user's messages
        user_chunks = []
        for chunk in all_chunks:
            metadata = chunk.get("metadata", {})
            # Check if chunk contains messages from this user
            # (This assumes message_ids are stored in chunk metadata)
            # You might need to cross-reference with message_storage
            # to check author_id

            # For now, simple approach: include all and let LLM handle it
            # TODO: Implement proper user filtering
            user_chunks.append(chunk)

        return user_chunks[:top_k]

    def _build_emulation_prompt(
        self,
        target_user_name: str,
        context: str,
        style_profile: Dict,
        example_messages: List[Dict]
    ) -> str:
        """
        Build prompt for LLM to emulate user.

        Learning: Style transfer prompting!
        We describe the style and give examples.
        """
        prompt_parts = [
            f"You are emulating how {target_user_name} would respond in Discord.",
            "",
            "STYLE PROFILE:",
        ]

        # Add style characteristics
        prompt_parts.append(f"- Message length: ~{style_profile['avg_message_length']} characters")
        prompt_parts.append(f"- Punctuation style: {style_profile['punctuation_style']}")
        prompt_parts.append(f"- Capitalization: {style_profile['capitalization_style']}")
        prompt_parts.append(f"- Formality: {style_profile['formality_score']:.1f}/1.0 (0=casual, 1=formal)")

        if style_profile['common_phrases']:
            prompt_parts.append(f"- Common phrases: {', '.join(style_profile['common_phrases'][:5])}")

        if style_profile['signature_phrases']:
            prompt_parts.append(f"- Signature phrases: {', '.join(style_profile['signature_phrases'])}")

        if style_profile['emoji_usage'] > 0.2:
            prompt_parts.append(f"- Uses emoji frequently ({style_profile['emoji_usage']:.0%} of messages)")

        # Add example messages
        if example_messages:
            prompt_parts.append("")
            prompt_parts.append("EXAMPLE MESSAGES from this user:")
            for i, msg in enumerate(example_messages[:3], 1):
                content = msg.get("content", "")[:200]
                prompt_parts.append(f"{i}. {content}")

        # Add context
        prompt_parts.append("")
        prompt_parts.append(f"CONTEXT/QUESTION: {context}")
        prompt_parts.append("")
        prompt_parts.append(f"Respond as {target_user_name} would (match their style, vocabulary, and tone):")

        return "\n".join(prompt_parts)

    def _calculate_confidence(
        self,
        user_messages: List[Dict],
        relevant_messages: List[Dict]
    ) -> float:
        """
        Calculate confidence in emulation.

        More messages = higher confidence
        More relevant context = higher confidence
        """
        # Base confidence on message count
        message_count = len(user_messages)
        base_confidence = min(message_count / 100, 0.8)  # Cap at 0.8

        # Boost if we found relevant context
        if relevant_messages:
            base_confidence += 0.2

        return min(base_confidence, 1.0)
```

### Step 12.3: Add Emulation Commands to Chatbot Cog

Add to `bot/cogs/chatbot.py`:

```python
@commands.command(name='emulate')
async def emulate(self, ctx, target_user: discord.Member, *, context: str):
    """
    Emulate how a user would respond.

    Usage:
        !emulate @Alice what do you think about the new API design?
        !emulate @Bob should we go with Python or JavaScript?

    Learning: This is ethically sensitive! Use responsibly.
    """
    try:
        # Check permissions (optional: only allow for specific roles)
        # For now, anyone can use it

        async with ctx.typing():
            # Initialize emulation service
            from ai.user_emulation import UserEmulationService
            from storage.messages import MessageStorage

            message_storage = MessageStorage()
            emulation_service = UserEmulationService(
                message_storage,
                self.conversational_rag.rag.memory,  # Access chunked memory
                self.conversational_rag.ai
            )

            # Emulate
            result = await emulation_service.emulate_response(
                target_user_id=str(target_user.id),
                target_user_name=target_user.display_name,
                context=context,
                channel_id=str(ctx.channel.id)
            )

            # Create embed
            if result["confidence"] < 0.3:
                await ctx.send(result["response"])
                return

            embed = discord.Embed(
                title=f"🎭 Emulating {target_user.display_name}",
                description=result["response"],
                color=discord.Color.purple()
            )

            # Add confidence meter
            confidence_bar = "█" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
            embed.add_field(
                name="Confidence",
                value=f"{confidence_bar} {result['confidence']:.0%}",
                inline=False
            )

            # Add style info
            if result.get("style_profile"):
                style = result["style_profile"]
                embed.add_field(
                    name="Style Profile",
                    value=(
                        f"Formality: {style['formality_score']:.1f}/1.0\n"
                        f"Punctuation: {style['punctuation_style']}\n"
                        f"Emoji usage: {style['emoji_usage']:.0%}"
                    ),
                    inline=True
                )

            embed.set_footer(text="⚠️ This is an emulation, not the actual user")

            await ctx.send(embed=embed)

    except Exception as e:
        self.logger.error(f"Error in emulate: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")

@commands.command(name='analyze_user')
async def analyze_user(self, ctx, target_user: discord.Member):
    """
    Analyze a user's communication style.

    Usage: !analyze_user @Alice
    """
    try:
        from ai.user_emulation import UserEmulationService
        from storage.messages import MessageStorage

        message_storage = MessageStorage()
        all_messages = message_storage.load_channel_messages(str(ctx.channel.id))
        user_messages = [
            msg for msg in all_messages
            if msg.get("author_id") == str(target_user.id)
        ]

        if len(user_messages) < 5:
            await ctx.send(f"Not enough messages from {target_user.display_name} to analyze.")
            return

        from ai.user_style_analyzer import UserStyleAnalyzer
        analyzer = UserStyleAnalyzer()
        style = analyzer.analyze_user_style(user_messages)

        # Create embed
        embed = discord.Embed(
            title=f"📊 Style Analysis: {target_user.display_name}",
            color=discord.Color.blue()
        )

        embed.add_field(
            name="Message Stats",
            value=(
                f"Messages analyzed: {len(user_messages)}\n"
                f"Avg length: {style['avg_message_length']} chars"
            ),
            inline=False
        )

        embed.add_field(
            name="Communication Style",
            value=(
                f"Formality: {style['formality_score']:.1f}/1.0\n"
                f"Punctuation: {style['punctuation_style']}\n"
                f"Capitalization: {style['capitalization_style']}\n"
                f"Emoji usage: {style['emoji_usage']:.0%}"
            ),
            inline=False
        )

        if style['common_phrases']:
            embed.add_field(
                name="Common Phrases",
                value=", ".join(style['common_phrases'][:10]),
                inline=False
            )

        if style['signature_phrases']:
            embed.add_field(
                name="Signature Phrases",
                value=", ".join(style['signature_phrases']),
                inline=False
            )

        await ctx.send(embed=embed)

    except Exception as e:
        self.logger.error(f"Error in analyze_user: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")
```

---

## Usage Examples

### Emulate User Response

```
!emulate @Alice what do you think about the new API design?

Bot: 🎭 [Emulating Alice]
     hmm i think the REST endpoints look pretty clean! but ngl we might
     need better error handling, esp for edge cases. just my 2 cents tho 🤔

     Confidence: ████████░░ 80%
     Style: Casual, uses "ngl", "imo", emoji frequently
```

### Analyze User Style

```
!analyze_user @Bob

Bot: 📊 Style Analysis: Bob
     Messages analyzed: 247
     Avg length: 78 characters

     Communication Style:
     - Formality: 0.7/1.0
     - Punctuation: formal
     - Capitalization: proper
     - Emoji usage: 5%

     Common Phrases: i think, we should, makes sense, good point
     Signature Phrases: just my opinion, fwiw
```

---

## Ethical Considerations

⚠️ **IMPORTANT**: User emulation is powerful but raises ethical concerns:

1. **Consent**: Consider requiring user opt-in for emulation
2. **Transparency**: Always mark emulated responses clearly
3. **No Impersonation**: Never use for deception
4. **Respect Privacy**: Don't emulate private/sensitive content
5. **Moderation**: Allow users to disable emulation of themselves

### Recommended Safeguards

```python
# Add to config.py
EMULATION_ENABLED: bool = os.getenv("EMULATION_ENABLED", "False").lower() == "true"
EMULATION_REQUIRES_PERMISSION: bool = True
EMULATION_BLACKLIST: List[str] = []  # User IDs who opted out

# In command:
if not Config.EMULATION_ENABLED:
    await ctx.send("Emulation mode is disabled on this server.")
    return

if str(target_user.id) in Config.EMULATION_BLACKLIST:
    await ctx.send(f"{target_user.display_name} has opted out of emulation.")
    return
```

---

## Common Pitfalls - Phase 12

1. **Too few messages**: Need 20+ for good emulation
2. **No filtering**: Emulating ALL users without permission
3. **Missing disclaimers**: Users think it's the real person
4. **Overfitting**: Repeating exact phrases instead of style
5. **Context mismatch**: Emulating user on topics they never discussed

## Debugging Tips - Phase 12

- **Test with known users**: Emulate yourself first
- **Check message counts**: Log how many messages found
- **Compare styles**: Show style analysis to user for validation
- **Monitor confidence**: Low confidence = more data needed

## Performance Considerations - Phase 12

- **Cache style profiles**: Don't recompute every time
- **Limit message analysis**: 100-200 most recent messages sufficient
- **Token usage**: Style prompts are long, watch costs

---

## Next Steps

This gives you **user emulation** that:
- ✅ Analyzes individual communication styles
- ✅ Generates responses in user's voice
- ✅ Shows confidence levels
- ✅ Respects ethical boundaries

Next up: **Phase 13 - Debate & Rhetoric Analyzer** 🎓
