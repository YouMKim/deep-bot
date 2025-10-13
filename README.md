# Deep Bot 🤖

A secure Discord bot built with Python, featuring OpenAI integration and designed to run on Raspberry Pi.

## 🔐 Security Features

- Environment variable management with `.env` files
- Secure configuration validation
- Proper `.gitignore` to prevent credential leaks
- Input validation and error handling
- Debug mode controls

## 📁 Project Structure

```
deep-bot/
├── bot.py                 # Main bot file
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── env.example          # Environment variables template
├── cogs/                # Bot command modules
│   ├── __init__.py
│   └── basic.py         # Basic commands
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd deep-bot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

1. Copy the example environment file:
   ```bash
   copy env.example .env
   ```

2. Edit `.env` file with your actual credentials:
   ```env
   DISCORD_TOKEN=your_actual_discord_token
   DISCORD_CLIENT_ID=your_actual_client_id
   DISCORD_GUILD_ID=your_server_id
   OPENAI_API_KEY=your_actual_openai_key
   BOT_OWNER_ID=your_discord_user_id
   ```

### 5. Run the Bot

```bash
python bot.py
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | ✅ | Your Discord bot token |
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |
| `DISCORD_CLIENT_ID` | ❌ | Bot client ID for invite links |
| `DISCORD_GUILD_ID` | ❌ | Specific server ID (optional) |
| `BOT_OWNER_ID` | ❌ | Your Discord user ID |
| `BOT_PREFIX` | ❌ | Command prefix (default: !) |
| `DEBUG_MODE` | ❌ | Enable debug mode (default: False) |
| `LOG_LEVEL` | ❌ | Logging level (default: INFO) |

### Getting Discord Bot Token

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section
4. Click "Add Bot"
5. Copy the token (keep it secret!)

### Getting OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new secret key
5. Copy the key (keep it secret!)

## 🛡️ Security Best Practices

1. **Never commit `.env` files** - They're in `.gitignore`
2. **Use environment variables** - Never hardcode secrets
3. **Validate configuration** - Bot won't start with missing required vars
4. **Use virtual environments** - Isolate dependencies
5. **Regular updates** - Keep dependencies updated
6. **Monitor logs** - Check `bot.log` for issues

## 🏗️ Development

### Adding New Commands

1. Create a new file in `cogs/` directory
2. Follow the pattern in `cogs/basic.py`
3. Load the cog in `bot.py` setup_hook method

### Example Cog Structure

```python
from discord.ext import commands

class MyCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.command()
    async def mycommand(self, ctx):
        await ctx.send("Hello!")

async def setup(bot):
    await bot.add_cog(MyCog(bot))
```

## 🍓 Raspberry Pi Setup

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+ (if not already installed)
sudo apt install python3 python3-pip python3-venv -y

# Install Git (if not already installed)
sudo apt install git -y
```

### Running on Pi

1. Follow the quick start guide above
2. Consider using a process manager like `systemd` for production
3. Monitor resource usage with `htop`

### Systemd Service (Optional)

Create `/etc/systemd/system/deep-bot.service`:

```ini
[Unit]
Description=Deep Discord Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/deep-bot
Environment=PATH=/home/pi/deep-bot/venv/bin
ExecStart=/home/pi/deep-bot/venv/bin/python bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable deep-bot
sudo systemctl start deep-bot
```

## 📝 Logging

- Logs are written to `bot.log`
- Console output shows real-time logs
- Log level can be configured in `.env`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter issues:

1. Check the logs in `bot.log`
2. Verify your `.env` configuration
3. Ensure all dependencies are installed
4. Check Discord bot permissions
5. Verify OpenAI API key is valid

---

**Remember: Keep your tokens and API keys secure! Never share them publicly.**
