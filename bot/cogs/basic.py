"""
Basic commands cog for the Discord bot.
Contains simple utility and fun commands.
"""

import discord
from discord.ext import commands
from discord.ext.commands import cooldown, BucketType
import asyncio
import re
from typing import Optional
from ai.service import AIService
from ai.tracker import UserAITracker
from config import Config


class Basic(commands.Cog):
    """Basic commands for the bot."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config
        provider = getattr(self.config, 'EVALUATE_PROVIDER', None) or self.config.AI_DEFAULT_PROVIDER
        self.ai_service = AIService(provider_name="openai")
        self.evaluate_ai_service = AIService(provider_name=provider)
        self.ai_tracker = UserAITracker()

    @commands.command(name="ping", help="Check bot latency")
    async def ping(self, ctx):
        """Check the bot's latency."""
        latency = round(self.bot.latency * 1000)
        await ctx.send(f"🏓 Pong! Latency: {latency}ms")

    @commands.command(name="hello", help="Say hello to the bot")
    async def hello(self, ctx):
        """Say hello to the bot."""
        user_name = ctx.author.display_name
        
        try:
            prompt = f"Say hello back to {user_name} in a casual way but make sure to make user feel bad for wasting electricity and resources for this simple greeting. Keep it short (1-2 sentences)."
            
            result = await self.ai_service.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=1.7,
            )
            
            embed = discord.Embed(
                title="Hello",
                description=result["content"],
                color=discord.Color.blue()
            )
            
            embed.set_footer(text=f"Cost: ${result['cost']:.6f} | Model: {result['model']}")
            
            await ctx.send(embed=embed)
            
            # Track AI usage (simple - just cost and tokens)
            self.ai_tracker.log_ai_usage(
                user_display_name=user_name,
                cost=result['cost'],
                tokens_total=result['tokens_total']
            )
            
        except Exception as e:
            await ctx.send(f"👋 Hello {ctx.author.mention}! (AI greeting failed: {e})")

    @commands.command(name="info", help="Get bot information")
    async def info(self, ctx):
        """Get bot information."""
        embed = discord.Embed(
            title="🤖 Bot Information",
            description="A Discord bot built with discord.py",
            color=discord.Color.blue(),
        )
        embed.add_field(name="Guilds", value=len(self.bot.guilds), inline=True)
        embed.add_field(name="Users", value=len(self.bot.users), inline=True)
        embed.add_field(
            name="Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True
        )
        embed.add_field(name="Prefix", value=self.bot.command_prefix, inline=True)
        embed.add_field(name="Debug Mode", value=self.bot.debug_mode, inline=True)

        await ctx.send(embed=embed)

    @commands.command(name="echo", help="Echo a message")
    async def echo(self, ctx, *, message):
        """Echo a message."""
        await ctx.send(f"📢 {message}")

    @commands.command(name="server", help="Get server information")
    @commands.guild_only()
    async def server(self, ctx):
        """Get information about the current server."""
        guild = ctx.guild

        embed = discord.Embed(title=f"📊 {guild.name}", color=discord.Color.green())
        embed.add_field(name="Members", value=guild.member_count, inline=True)
        embed.add_field(name="Channels", value=len(guild.channels), inline=True)
        embed.add_field(name="Roles", value=len(guild.roles), inline=True)
        embed.add_field(
            name="Created", value=guild.created_at.strftime("%Y-%m-%d"), inline=True
        )
        embed.add_field(
            name="Owner",
            value=guild.owner.mention if guild.owner else "Unknown",
            inline=True,
        )

        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)

        await ctx.send(embed=embed)
    
    @commands.command(name="mystats", help="View your AI usage stats and social credit")
    async def mystats(self, ctx):
        user_name = ctx.author.display_name
        user_id = str(ctx.author.id)
        
        # Get AI usage stats
        usage_stats = self.ai_tracker.get_user_stats(user_name)
        
        from config import Config
        social_credit_score = None
        if Config.SOCIAL_CREDIT_ENABLED:
            from ai.social_credit import SocialCreditManager
            sc_manager = SocialCreditManager()
            sc_stats = await sc_manager.get_user_stats(user_id)
            if sc_stats:
                social_credit_score = sc_stats['social_credit_score']
            else:
                # Initialize if doesn't exist
                social_credit_score = await sc_manager.get_or_initialize_score(user_id, user_name)
        
        embed = discord.Embed(
            title=f"📊 Clanker Stats for {user_name}",
            color=discord.Color.green()
        )
        
        # Social Credit (new behavior-based system)
        if social_credit_score is not None:
            embed.add_field(
                name="Social Credit Score",
                value=f"{social_credit_score}",
                inline=True
            )
        
        # AI Usage Stats (if available)
        if usage_stats:
            # Lifetime Spending
            embed.add_field(
                name="Money Wasted So Far",
                value=f"${usage_stats['lifetime_cost']:.6f}",
                inline=True
            )
            
            # Lifetime Tokens
            embed.add_field(
                name="Tokens Used So Far",
                value=f"{usage_stats['lifetime_tokens']:,}",
                inline=True
            )
        else:
            embed.add_field(
                name="AI Usage",
                value="No usage recorded yet",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='help', help='Show available commands for regular users')
    async def help_command(self, ctx):
        """Show available commands for regular users."""
        embed = discord.Embed(
            title="📚 Bot Commands",
            description="Available commands for regular users",
            color=discord.Color.blue()
        )
        
        # Get all commands and filter out admin-only ones
        admin_commands = {
            'whoami', 'check_blacklist', 'reload_blacklist', 'load_channel', 'load_server',
            'check_storage', 'checkpoint_info', 'chunk_status', 'reset_chunk_checkpoint',
            'rechunk', 'rag_settings', 'rag_set', 'rag_reset', 'rag_enable_all',
            'compare_rag', 'ai_provider', 'chatbot_mode'
        }
        
        # Group commands by cog
        commands_by_cog = {}
        for command in self.bot.commands:
            # Skip admin commands and the help_admin command
            if command.name in admin_commands or command.name == 'help_admin':
                continue
            
            # Skip if command is owner-only
            if command.checks:
                is_owner_only = any(
                    hasattr(check, '__name__') and 'is_owner' in str(check)
                    for check in command.checks
                )
                if is_owner_only:
                    continue
            
            cog_name = command.cog.qualified_name if command.cog else "Other"
            if cog_name not in commands_by_cog:
                commands_by_cog[cog_name] = []
            commands_by_cog[cog_name].append(command)
        
        # Add commands to embed
        for cog_name in sorted(commands_by_cog.keys()):
            commands_list = commands_by_cog[cog_name]
            if commands_list:
                value = "\n".join([
                    f"`{self.bot.command_prefix}{cmd.name}` - {cmd.help or 'No description'}"
                    for cmd in sorted(commands_list, key=lambda x: x.name)
                ])
                embed.add_field(
                    name=cog_name,
                    value=value,
                    inline=False
                )
        
        embed.set_footer(text=f"Use {self.bot.command_prefix}help_admin for admin commands")
        await ctx.send(embed=embed)
    
    @commands.command(name='help_admin', help='Show available admin commands (Admin only)')
    @commands.is_owner()
    async def help_admin(self, ctx):
        """Show available admin commands."""
        embed = discord.Embed(
            title="🔐 Admin Commands",
            description="Available commands for bot administrators",
            color=discord.Color.red()
        )
        
        # Admin commands grouped by category
        admin_categories = {
            "Bot Management": [
                'whoami',
                'ai_provider',
            ],
            "Blacklist": [
                'check_blacklist',
                'reload_blacklist',
            ],
            "Message Loading": [
                'load_channel',
                'load_server',
                'check_storage',
            ],
            "Chunking": [
                'chunk_status',
                'checkpoint_info',
                'reset_chunk_checkpoint',
                'rechunk',
            ],
            "RAG Settings": [
                'rag_settings',
                'rag_set',
                'rag_reset',
                'rag_enable_all',
                'compare_rag',
            ],
            "Chatbot": [
                'chatbot_mode',
            ],
        }
        
        for category, command_names in admin_categories.items():
            commands_list = []
            for cmd_name in command_names:
                cmd = self.bot.get_command(cmd_name)
                if cmd:
                    commands_list.append(f"`{self.bot.command_prefix}{cmd.name}` - {cmd.help or 'No description'}")
            
            if commands_list:
                embed.add_field(
                    name=category,
                    value="\n".join(commands_list),
                    inline=False
                )
        
        embed.set_footer(text=f"Use {self.bot.command_prefix}help for regular user commands")
        await ctx.send(embed=embed)

    def _create_fact_checking_prompt(self, claim: str) -> str:
        """
        Create the system prompt for fact-checking evaluation.
        
        Args:
            claim: The claim to be evaluated
            
        Returns:
            System prompt string
        """
        return """You are a fact-checking assistant. Evaluate claims concisely (aim for 3-5 sentences).

INSTRUCTIONS:
1. Search the web for information about the claim. Use reputable sources.
2. Cite sources with URLs: [Source Name](URL) or include URLs directly.
3. Provide brief analysis: Is the claim accurate? What does the evidence show?
4. End with ratings in this exact format:
   TRUTHFULNESS: [0-100]%
   EVIDENCE_ALIGNMENT: [0-100]%
5. List sources as URLs at the end.

Be concise, objective, and cite your sources. Keep response under 500 words."""

    def _extract_ratings(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """
        Extract truthfulness and evidence alignment ratings from LLM response.
        
        Args:
            text: The LLM response text
            
        Returns:
            Tuple of (truthfulness_rating, evidence_alignment_rating) or (None, None) if not found
        """
        truthfulness = None
        evidence_alignment = None
        
        # Look for "TRUTHFULNESS: X%" pattern
        truth_match = re.search(r'TRUTHFULNESS:\s*(\d+)%', text, re.IGNORECASE)
        if truth_match:
            truthfulness = int(truth_match.group(1))
        
        # Look for "EVIDENCE_ALIGNMENT: X%" pattern
        evidence_match = re.search(r'EVIDENCE_ALIGNMENT:\s*(\d+)%', text, re.IGNORECASE)
        if evidence_match:
            evidence_alignment = int(evidence_match.group(1))
        
        return truthfulness, evidence_alignment

    def _split_text_at_sentence_boundary(self, text: str, max_length: int) -> list[str]:
        """
        Split text into chunks at sentence boundaries, preserving paragraph structure.
        
        Preserves:
        - Paragraph breaks (double newlines \n\n)
        - Single newlines within paragraphs
        - Sentence punctuation and spacing
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks with preserved formatting
        """
        import re
        chunks = []
        current_chunk = ""
        
        # Split on sentence endings (period, exclamation, question mark followed by space/newline)
        # This preserves the punctuation and whitespace including paragraph breaks
        sentences = re.split(r'([.!?][\s\n]+)', text)
        
        def preserve_formatting(chunk: str) -> str:
            """Preserve paragraph breaks while cleaning trailing spaces."""
            # Remove trailing spaces/tabs but preserve all newlines (including paragraph breaks \n\n)
            # This keeps paragraph structure intact while cleaning up trailing whitespace
            return chunk.rstrip(' \t')  # Only strip spaces/tabs, keep newlines
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] if i < len(sentences) else ""
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            if not full_sentence.strip():
                continue
            
            # If adding this sentence would exceed limit, save current chunk and start new one
            if current_chunk and len(current_chunk) + len(full_sentence) > max_length:
                # Preserve paragraph structure - keep newlines, remove trailing spaces
                if chunks:
                    chunks.append(preserve_formatting(current_chunk))
                else:
                    chunks.append(current_chunk.strip())
                current_chunk = full_sentence.lstrip()  # Remove leading whitespace from new chunk start
            else:
                current_chunk += full_sentence
        
        # Add remaining chunk - preserve all formatting
        if current_chunk.strip():
            if chunks:
                chunks.append(preserve_formatting(current_chunk))
            else:
                chunks.append(current_chunk.strip())
        
        return chunks

    def _remove_urls_from_text(self, text: str, urls_to_remove: list[str]) -> str:
        """
        Remove URLs from text to avoid duplication (since URLs are shown in separate sources embed).
        Also cleans up orphaned text like "Source:" or "and" that was around the URLs.
        
        Args:
            text: Text that may contain URLs
            urls_to_remove: List of URLs to remove from the text
            
        Returns:
            Text with URLs removed and cleaned up
        """
        import re
        cleaned_text = text
        
        # Remove markdown links [text](url) where URL matches - remove entire link
        for url in urls_to_remove:
            # Escape special regex characters in URL
            escaped_url = re.escape(url)
            # Remove markdown links: [anything](url) - remove the whole thing
            cleaned_text = re.sub(rf'\[([^\]]+)\]\({escaped_url}\)', '', cleaned_text)
            # Remove plain URLs (with optional trailing punctuation)
            cleaned_text = re.sub(rf'{escaped_url}[.,;:!?)]*', '', cleaned_text)
        
        # Clean up orphaned text patterns that commonly appear around URLs
        # Remove "Source:" or "Sources:" followed by nothing, dashes, or "and"
        cleaned_text = re.sub(r'Source[s]?:\s*[-–—]?\s*(and\s*[-–—]?\s*)?', '', cleaned_text, flags=re.IGNORECASE)
        # Remove orphaned "and" connectors (standalone or after punctuation)
        cleaned_text = re.sub(r'(^|\n|\.|,|:)\s*and\s*[-–—]?\s*', r'\1 ', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove standalone dashes, bullets, or separators
        cleaned_text = re.sub(r'\s*[-–—•]\s*$', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'^\s*[-–—•]\s*', '', cleaned_text, flags=re.MULTILINE)
        # Remove "Source:" or "Sources:" at end of sentences or lines
        cleaned_text = re.sub(r'\.\s*Source[s]?:\s*', '. ', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'Source[s]?:\s*$', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove patterns like "Source: and -" or "Sources: -"
        cleaned_text = re.sub(r'Source[s]?:\s*and\s*[-–—]', '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace that might result from URL removal
        # Preserve newlines: only collapse multiple spaces/tabs, not newlines
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Multiple spaces/tabs to single space (preserves newlines)
        cleaned_text = re.sub(r'[ \t]+$', '', cleaned_text, flags=re.MULTILINE)  # Remove trailing spaces
        cleaned_text = re.sub(r'^[ \t]+', '', cleaned_text, flags=re.MULTILINE)  # Remove leading spaces
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Multiple newlines to double (preserve paragraph breaks)
        # Remove lines that are just dashes, bullets, or punctuation
        cleaned_text = re.sub(r'^\s*[-–—•]\s*$', '', cleaned_text, flags=re.MULTILINE)
        # Clean up double punctuation
        cleaned_text = re.sub(r'\.\s*\.', '.', cleaned_text)
        cleaned_text = re.sub(r',\s*,', ',', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def _extract_urls(self, text: str) -> list[str]:
        """
        Extract URLs from the LLM response.
        
        Args:
            text: The LLM response text
            
        Returns:
            List of URLs found in the text
        """
        urls = []
        
        # First, extract markdown links [text](url) - get the URL part
        markdown_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
        for match in re.finditer(markdown_pattern, text):
            url = match.group(2)
            urls.append(url)
        
        # Then extract plain URLs (http/https) - but skip if already in markdown links
        url_pattern = r'https?://[^\s\)]+'
        for match in re.finditer(url_pattern, text):
            url = match.group(0)
            # Check if this URL is part of a markdown link we already captured
            is_in_markdown = False
            for markdown_match in re.finditer(markdown_pattern, text):
                if url in markdown_match.group(0):
                    is_in_markdown = True
                    break
            if not is_in_markdown:
                urls.append(url)
        
        # Deduplicate
        all_urls = list(set(urls))
        
        # Clean up URLs (remove trailing punctuation)
        cleaned_urls = []
        for url in all_urls:
            # Remove trailing punctuation that might have been included
            url = url.rstrip('.,;:!?)')
            if url.startswith('http'):
                cleaned_urls.append(url)
        
        return cleaned_urls[:10]  # Limit to 10 sources

    async def _summarize_evaluation(self, evaluation_text: str, claim: str, user_id: str) -> str:
        """
        Summarize a very long evaluation response to fit within embed limits.
        
        Args:
            evaluation_text: Long evaluation text to summarize
            claim: Original claim being evaluated
            user_id: User ID for social credit tone injection
            
        Returns:
            Summarized evaluation text
        """
        summarize_prompt = f"""The user asked to evaluate this claim: "{claim[:200]}"

The AI generated this detailed evaluation, but it's too long for Discord. Please create a concise summary that:
- Preserves the truthfulness and evidence alignment ratings
- Keeps the most important analysis points
- Maintains key evidence and sources mentioned
- Aim for 3-5 sentences covering the essential evaluation

Original evaluation:
{evaluation_text[:4000]}

Provide a concise summary:"""
        
        try:
            summary_result = await self.evaluate_ai_service.generate(
                prompt=summarize_prompt,
                max_tokens=500,  # Enough for a good summary
                temperature=0.3,  # Lower temperature for more focused summary
                user_id=user_id,
                user_display_name=None
            )
            
            summary = summary_result.get('content', '').strip()
            if summary and len(summary) < len(evaluation_text):
                return summary
            else:
                # If summarization didn't help, return original (will be split)
                return evaluation_text
        except Exception as e:
            self.logger.error(f"Error summarizing evaluation: {e}", exc_info=True)
            return evaluation_text  # Return original if summarization fails

    def _get_rating_color(self, rating: int) -> discord.Color:
        """
        Get Discord embed color based on rating.
        
        Args:
            rating: Rating value (0-100)
            
        Returns:
            Discord Color object
        """
        if rating >= 70:
            return discord.Color.green()
        elif rating >= 40:
            return discord.Color.gold()
        else:
            return discord.Color.red()

    @commands.command(
        name="evaluate",
        help="Fact-check a message by replying to it (searches web and cites sources)"
    )
    @cooldown(rate=3, per=60, type=BucketType.user)
    async def evaluate(self, ctx):
        """
        Evaluate a message for fact-checking.
        Must be used as a reply to a message.
        """
        # Check if command is used as a reply
        if not ctx.message.reference:
            await ctx.send("❌ Please reply to a message to evaluate it. Use `!evaluate` as a reply.")
            return
        
        # Check if evaluate is enabled
        if not getattr(self.config, 'EVALUATE_ENABLED', True):
            await ctx.send("❌ The evaluate command is currently disabled.")
            return
        
        # Get the replied message
        try:
            replied_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        except discord.NotFound:
            await ctx.send("❌ The message you're replying to was not found. It may have been deleted.")
            return
        except discord.Forbidden:
            await ctx.send("❌ I don't have permission to access that message.")
            return
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error fetching replied message: {e}", exc_info=True)
            await ctx.send("❌ An error occurred while fetching the message.")
            return
        
        # Extract claim from replied message
        claim = replied_message.content
        if not claim or not claim.strip():
            await ctx.send("❌ The message you're replying to appears to be empty.")
            return
        
        # Send initial status message
        status_msg = await ctx.send("🔍 Evaluating claim and searching for sources...")
        
        try:
            # Get user info for social credit tone injection
            user_id = str(ctx.author.id)
            user_display_name = ctx.author.display_name
            
            # Create fact-checking prompt
            system_prompt = self._create_fact_checking_prompt(claim)
            user_prompt = f"""Please evaluate the following claim:

"{claim}"

Search the web for relevant information, cite your sources with URLs, and provide your evaluation with ratings."""

            # Get configuration values
            max_tokens = getattr(self.config, 'EVALUATE_MAX_TOKENS', 1500)
            temperature = getattr(self.config, 'EVALUATE_TEMPERATURE', 0.3)
            
            # Generate evaluation using LLM
            result = await self.evaluate_ai_service.generate(
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                user_id=user_id,
                user_display_name=user_display_name,
                system_prompt=system_prompt
            )
            
            evaluation_text = result['content']
            
            # Extract ratings
            truthfulness, evidence_alignment = self._extract_ratings(evaluation_text)
            
            # Apply social credit penalty if claim is false
            if truthfulness is not None and truthfulness < 40 and Config.SOCIAL_CREDIT_ENABLED:
                try:
                    from ai.social_credit import SocialCreditManager
                    sc_manager = SocialCreditManager()
                    
                    # Get the author of the evaluated message (not the evaluator)
                    evaluated_user_id = str(replied_message.author.id)
                    evaluated_user_name = replied_message.author.display_name
                    
                    # Apply -50 penalty for spreading misinformation
                    new_score = await sc_manager.update_score(
                        evaluated_user_id,
                        -50,
                        "spreading_misinformation",
                        evaluated_user_name
                    )
                    
                    # Announce the penalty with emojis
                    penalty_message = (
                        f"🚨 **SPREADING MISINFORMATION??** 🚨\n"
                        f"{replied_message.author.mention} **-50 SOCIAL CREDIT SCORE**\n"
                        f"New score: **{new_score}**"
                    )
                    await ctx.send(penalty_message)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error applying social credit penalty: {e}", exc_info=True)
                    # Continue with evaluation even if penalty fails
            
            # Extract URLs (limit to top 3) BEFORE removing them from text
            urls = self._extract_urls(evaluation_text)[:3]
            
            # Remove URLs from evaluation text since they'll be shown in separate sources embed
            evaluation_text = self._remove_urls_from_text(evaluation_text, urls)
            
            # Determine embed color based on truthfulness (or default to blue)
            if truthfulness is not None:
                embed_color = self._get_rating_color(truthfulness)
            else:
                embed_color = discord.Color.blue()
            
            # Prepare claim preview
            claim_preview = claim[:500] + "..." if len(claim) > 500 else claim
            
            # Summarize if analysis is very long (before splitting)
            SUMMARIZE_THRESHOLD = 6000  # Summarize if > 6000 chars
            original_length = len(evaluation_text)
            if original_length > SUMMARIZE_THRESHOLD:
                try:
                    evaluation_text = await self._summarize_evaluation(evaluation_text, claim, user_id)
                    self.logger.info(f"Summarized evaluation from {original_length} to {len(evaluation_text)} chars")
                except Exception as e:
                    self.logger.error(f"Failed to summarize evaluation: {e}", exc_info=True)
                    # Continue with original text if summarization fails
            
            # Split analysis text into chunks if needed (max 4000 chars per embed description)
            max_analysis_per_embed = 4000
            analysis_chunks = self._split_text_at_sentence_boundary(evaluation_text, max_analysis_per_embed)
            
            # Create first embed with claim, ratings, and first part of analysis
            embeds = []
            first_embed = discord.Embed(
                title="🔍 Fact-Check Evaluation",
                color=embed_color,
                timestamp=discord.utils.utcnow()
            )
            
            # Add claim field
            first_embed.add_field(
                name="Claim",
                value=f"**{replied_message.author.display_name}**: {claim_preview}",
                inline=False
            )
            
            # Add ratings
            if truthfulness is not None:
                truthfulness_label = f"{truthfulness}%"
                if truthfulness >= 70:
                    truthfulness_label += " - Mostly True"
                elif truthfulness >= 40:
                    truthfulness_label += " - Partially True"
                else:
                    truthfulness_label += " - Mostly False"
                
                first_embed.add_field(
                    name="Truthfulness",
                    value=truthfulness_label,
                    inline=True
                )
            else:
                first_embed.add_field(
                    name="Truthfulness",
                    value="Rating not found",
                    inline=True
                )
            
            if evidence_alignment is not None:
                evidence_label = f"{evidence_alignment}%"
                if evidence_alignment >= 70:
                    evidence_label += " - Strongly Supported"
                elif evidence_alignment >= 40:
                    evidence_label += " - Moderately Supported"
                else:
                    evidence_label += " - Weakly Supported"
                
                first_embed.add_field(
                    name="Evidence Alignment",
                    value=evidence_label,
                    inline=True
                )
            else:
                first_embed.add_field(
                    name="Evidence Alignment",
                    value="Rating not found",
                    inline=True
                )
            
            # Add first chunk of analysis as description (allows up to 4096 chars)
            if analysis_chunks:
                analysis_text = analysis_chunks[0]
                if len(analysis_text) > 4090:  # Leave buffer
                    analysis_text = analysis_text[:4090].rsplit(' ', 1)[0] + "..."
                first_embed.description = f"**Analysis:**\n{analysis_text}"
            
            # Add footer with metadata
            first_embed.set_footer(text=f"Model: {result['model']} | Cost: ${result['cost']:.6f}")
            embeds.append(first_embed)
            
            # Add additional analysis embeds if needed
            for i, chunk in enumerate(analysis_chunks[1:], 2):
                analysis_text = chunk
                if len(analysis_text) > 4090:  # Leave buffer
                    analysis_text = analysis_text[:4090].rsplit(' ', 1)[0] + "..."
                
                analysis_embed = discord.Embed(
                    title=f"🔍 Fact-Check Evaluation (Part {i})",
                    description=f"**Analysis (continued):**\n{analysis_text}",
                    color=embed_color,
                    timestamp=discord.utils.utcnow()
                )
                analysis_embed.set_footer(text=f"Model: {result['model']} | Cost: ${result['cost']:.6f}")
                embeds.append(analysis_embed)
            
            # Create sources embed (separate embed)
            if urls:
                sources_embed = discord.Embed(
                    title="🔍 Sources",
                    color=embed_color,
                    timestamp=discord.utils.utcnow()
                )
                
                sources_text = ""
                max_sources_length = 1020  # Leave buffer
                for url in urls:
                    # Truncate URL display text, but full URL still counts in markdown
                    if len(url) > 60:
                        url_display = url[:60] + "..."
                    else:
                        url_display = url
                    
                    # Calculate actual length: "• [display](full_url)\n"
                    url_line = f"• [{url_display}]({url})\n"
                    url_line_length = len(url_line)
                    
                    # Check if adding this URL would exceed limit
                    if len(sources_text) + url_line_length > max_sources_length:
                        break
                    sources_text += url_line
                
                sources_text = sources_text.rstrip()  # Remove trailing newline
                if sources_text:
                    sources_embed.add_field(
                        name="Sources",
                        value=sources_text,
                        inline=False
                    )
                    sources_embed.set_footer(text=f"Model: {result['model']} | Cost: ${result['cost']:.6f}")
                    embeds.append(sources_embed)
            
            # Send all embeds
            await status_msg.edit(content=None, embed=embeds[0])
            for embed in embeds[1:]:
                await ctx.send(embed=embed)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in evaluate command: {e}", exc_info=True)
            await status_msg.edit(content="❌ An error occurred while evaluating the claim. Please try again later.")
    
    @evaluate.error
    async def evaluate_error(self, ctx, error):
        """Handle errors for the evaluate command."""
        if isinstance(error, commands.CommandOnCooldown):
            minutes, seconds = divmod(int(error.retry_after), 60)
            embed = discord.Embed(
                title="⏰ Rate Limit Reached",
                description=(
                    f"Fact-checking is expensive!\n\n"
                    f"Please wait **{minutes}m {seconds}s** before trying again.\n\n"
                    f"This helps prevent API cost overruns."
                ),
                color=discord.Color.orange()
            )
            embed.set_footer(text="Limit: 3 evaluations per minute")
            await ctx.send(embed=embed)
        else:
            raise error


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(Basic(bot))
