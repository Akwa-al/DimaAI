# DimaAI - OSINT Discord Bot

A Discord bot that gathers open-source intelligence using multiple free tools and APIs. Supports natural language input with automatic intent detection powered by Groq LLM.

## What it does

- **Username OSINT** - Sherlock, Maigret, GitHub search
- **Email OSINT** - Holehe (account discovery), HaveIBeenPwned breach check
- **IP lookup** - geolocation, ISP, proxy/VPN detection
- **Domain recon** - certificate transparency via crt.sh, subdomain discovery
- **Phone lookup** - carrier info, validation
- **Web search** - DuckDuckGo dorking with smart query generation via Groq LLM

## Setup

1. Clone and install dependencies:

```bash
git clone https://github.com/Akwa-al/DimaAI.git
cd DimaAI
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

- `DISCORD_TOKEN` - from [Discord Developer Portal](https://discord.com/developers/applications)
- `GROQ_API_KEY` - free at [console.groq.com](https://console.groq.com)
- `HIBP_API_KEY` - optional, from [haveibeenpwned.com/API/Key](https://haveibeenpwned.com/API/Key)
- `GITHUB_TOKEN` - optional, increases GitHub API rate limits

3. Install OSINT tools (optional but recommended):

```bash
pip install sherlock-project maigret holehe
```

4. Run the bot:

```bash
python bot.py
```

## Usage

DM the bot or use `!osint <query>` in a server channel:

```
!osint @johndoe
!osint user@email.com
!osint 8.8.8.8
!osint example.com
```

The bot also understands natural language queries in DMs.

## Project Structure

```
bot.py              Main bot (OSINT + Discord)
recallai/           Discord Recall AI (message archiving + search)
requirements.txt    Python dependencies
.env.example        Environment variable template
```

## Requirements

- Python 3.10+
- discord.py 2.3+
- A Discord bot token with message content intent enabled
