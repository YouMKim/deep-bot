import discord

def format_discord_message(message: discord.Message) -> dict:
    return {
        'id': message.id,
        'content': message.content,
        'author': message.author.display_name,
        'author_id': message.author.id,
        'author_display_name': message.author.display_name,
        'channel_id': message.channel.id,
        'channel_name': message.channel.name,
        'guild_id': message.guild.id if message.guild else None,
        'guild_name': message.guild.name if message.guild else 'DM',
        'timestamp': message.created_at.isoformat(),
        'created_at': message.created_at.isoformat(),
        'is_bot': message.author.bot,
        'has_attachments': len(message.attachments) > 0,
        'message_type': 'reply' if message.reference else 'default'
    }

