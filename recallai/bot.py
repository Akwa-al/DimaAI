import asyncio
import aiohttp
import aiosqlite
import json
import os
import logging
import sqlite3
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
from contextlib import asynccontextmanager

import discord
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('discord_recall.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_OAUTH_BASE = "https://discord.com/oauth2/authorize"
DISCORD_TOKEN_URL = "https://discord.com/api/v10/oauth2/token"
DISCORD_USER_URL = "https://discord.com/api/v10/users/@me"
EMBEDDING_DIM = 384

CONFIG = {
    'bot_token': os.getenv('DISCORD_BOT_TOKEN'),
    'client_id': os.getenv('DISCORD_CLIENT_ID'),
    'client_secret': os.getenv('DISCORD_CLIENT_SECRET'),
    'redirect_uri': os.getenv('DISCORD_REDIRECT_URI'),
    'database_path': os.getenv('DATABASE_PATH', 'discord_recall.db'),
    'web_port': int(os.getenv('WEB_PORT', 8000)),
    'web_base_url': os.getenv('WEB_BASE_URL', 'http://localhost:8000'),
    'max_messages_per_user': int(os.getenv('MAX_MESSAGES_PER_USER', 50000)),
    'model_cache_dir': os.getenv('MODEL_CACHE_DIR', './models'),
}

required = ['bot_token', 'client_id', 'client_secret', 'redirect_uri']
missing = [k for k in required if not CONFIG[k]]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

Path(CONFIG['model_cache_dir']).mkdir(parents=True, exist_ok=True)

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS authorized_users (
    user_id TEXT PRIMARY KEY,
    username TEXT,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP,
    authorized_at TIMESTAMP,
    last_indexed_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    indexing_status TEXT DEFAULT 'pending',
    total_messages_indexed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_channels (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES authorized_users(user_id),
    channel_id TEXT,
    channel_name TEXT,
    channel_type TEXT,
    last_synced TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_messages (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES authorized_users(user_id),
    channel_id TEXT,
    author_id TEXT,
    author_name TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    edited_timestamp TIMESTAMP,
    attachments TEXT,
    embeds TEXT,
    mentions TEXT,
    reactions TEXT,
    referenced_message_id TEXT,
    is_bot BOOLEAN,
    embedding BLOB,
    has_embedding BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_files (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    message_id TEXT,
    filename TEXT,
    file_size INTEGER,
    mime_type TEXT,
    url TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS user_messages_fts USING fts5(
    content, author_name, tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS oauth_states (
    state TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    used BOOLEAN DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_user_messages_user ON user_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_user_messages_timestamp ON user_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_oauth_states_user ON oauth_states(user_id);
"""

app = FastAPI(title="Discord Recall AI - OAuth2", version="2.0.0")


@dataclass
class DiscordMessage:
    id: str
    channel_id: str
    author_id: str
    author_name: str
    content: str
    timestamp: datetime
    edited_timestamp: Optional[datetime] = None
    attachments: List[Dict] = None
    embeds: List[Dict] = None
    mentions: List[str] = None
    reactions: List[Dict] = None
    referenced_message_id: Optional[str] = None
    is_bot: bool = False


class EmbeddingService:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.model = None

    def initialize(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=self.cache_dir)
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None

    def get_embedding(self, text: str) -> np.ndarray:
        if not self.model:
            self.initialize()
        try:
            return self.model.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(EMBEDDING_DIM)


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(CREATE_TABLES)
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    @asynccontextmanager
    async def get_connection(self):
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        try:
            yield conn
        finally:
            await conn.close()

    async def store_oauth_token(self, user_id, username, access_token, refresh_token, expires_in):
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO authorized_users
                (user_id, username, access_token, refresh_token, token_expires_at,
                 authorized_at, indexing_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, username, access_token, refresh_token,
                  expires_at.isoformat(), datetime.now().isoformat(), 'pending'))
            await conn.commit()

    async def store_user_messages(self, user_id: str, messages: List[DiscordMessage]):
        async with self.get_connection() as conn:
            for msg in messages:
                try:
                    cursor = await conn.execute(
                        "SELECT id FROM user_messages WHERE id = ? AND user_id = ?",
                        (msg.id, user_id)
                    )
                    existing = await cursor.fetchone()

                    if existing:
                        await conn.execute("""
                            UPDATE user_messages SET
                                content = ?, edited_timestamp = ?, attachments = ?,
                                embeds = ?, mentions = ?, reactions = ?
                            WHERE id = ? AND user_id = ?
                        """, (
                            msg.content,
                            msg.edited_timestamp.isoformat() if msg.edited_timestamp else None,
                            json.dumps(msg.attachments) if msg.attachments else None,
                            json.dumps(msg.embeds) if msg.embeds else None,
                            json.dumps(msg.mentions) if msg.mentions else None,
                            json.dumps(msg.reactions) if msg.reactions else None,
                            msg.id, user_id
                        ))
                    else:
                        await conn.execute("""
                            INSERT INTO user_messages (
                                id, user_id, channel_id, author_id, author_name, content,
                                timestamp, edited_timestamp, attachments, embeds,
                                mentions, reactions, referenced_message_id, is_bot
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            msg.id, user_id, msg.channel_id, msg.author_id, msg.author_name,
                            msg.content, msg.timestamp.isoformat(),
                            msg.edited_timestamp.isoformat() if msg.edited_timestamp else None,
                            json.dumps(msg.attachments) if msg.attachments else None,
                            json.dumps(msg.embeds) if msg.embeds else None,
                            json.dumps(msg.mentions) if msg.mentions else None,
                            json.dumps(msg.reactions) if msg.reactions else None,
                            msg.referenced_message_id, msg.is_bot
                        ))
                        await conn.execute("""
                            INSERT INTO user_messages_fts (rowid, content, author_name)
                            SELECT id, content, author_name FROM user_messages WHERE id = ?
                        """, (msg.id,))

                    if msg.attachments:
                        for attachment in msg.attachments:
                            await conn.execute("""
                                INSERT OR REPLACE INTO user_files (
                                    id, user_id, message_id, filename, file_size, mime_type, url
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                attachment['id'], user_id, msg.id, attachment['filename'],
                                attachment.get('size', 0), attachment.get('content_type', 'unknown'),
                                attachment.get('url', '')
                            ))

                    if msg.content and len(msg.content.strip()) > 0:
                        embedding = embedding_service.get_embedding(msg.content)
                        if embedding is not None:
                            await conn.execute("""
                                UPDATE user_messages SET embedding = ?, has_embedding = 1
                                WHERE id = ?
                            """, (embedding.tobytes(), msg.id))

                except Exception as e:
                    logger.error(f"Error storing message {msg.id}: {e}")
                    continue

            await conn.commit()

    async def search_user_messages(self, user_id: str, query: str, limit: int = 50) -> List[Dict]:
        results = []
        query_embedding = embedding_service.get_embedding(query) if embedding_service else None

        async with self.get_connection() as conn:
            fts_cursor = await conn.execute("""
                SELECT m.id, m.content, m.author_name, m.timestamp, m.channel_id,
                       m.attachments, m.embedding, m.has_embedding, rank
                FROM user_messages_fts fts
                JOIN user_messages m ON m.id = fts.rowid
                WHERE m.user_id = ? AND user_messages_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (user_id, query, limit * 2))

            fts_results = await fts_cursor.fetchall()
            for row in fts_results:
                result = {
                    'id': row['id'], 'content': row['content'], 'author_name': row['author_name'],
                    'timestamp': row['timestamp'], 'channel_id': row['channel_id'],
                    'attachments': json.loads(row['attachments']) if row['attachments'] else [],
                    'score': 1.0 / (row['rank'] + 1), 'search_type': 'fts'
                }
                if query_embedding is not None and row['has_embedding'] and row['embedding']:
                    stored_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    similarity = np.dot(query_embedding, stored_embedding)
                    result['semantic_score'] = float(similarity)
                    result['score'] = (result['score'] + similarity) / 2
                results.append(result)

            if query_embedding is not None:
                semantic_cursor = await conn.execute("""
                    SELECT id, content, author_name, timestamp, channel_id, attachments, embedding
                    FROM user_messages WHERE user_id = ? AND has_embedding = 1
                    ORDER BY timestamp DESC LIMIT 1000
                """, (user_id,))
                for row in await semantic_cursor.fetchall():
                    if row['embedding']:
                        stored_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                        similarity = np.dot(query_embedding, stored_embedding)
                        if similarity > 0.5:
                            results.append({
                                'id': row['id'], 'content': row['content'],
                                'author_name': row['author_name'], 'timestamp': row['timestamp'],
                                'channel_id': row['channel_id'],
                                'attachments': json.loads(row['attachments']) if row['attachments'] else [],
                                'score': float(similarity), 'search_type': 'semantic'
                            })

        seen_ids = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x['score'], reverse=True):
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)
                if len(unique_results) >= limit:
                    break
        return unique_results


class UserIndexer:

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service

    async def index_user(self, user_id: str, access_token: str):
        try:
            logger.info(f"Starting index for user {user_id}")
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    "UPDATE authorized_users SET indexing_status = 'indexing' WHERE user_id = ?",
                    (user_id,)
                )
                await conn.commit()

            channels = await self._get_user_channels(access_token, user_id)

            total_messages = 0
            for channel in channels:
                if total_messages >= CONFIG['max_messages_per_user']:
                    break
                messages = await self._fetch_channel_messages(
                    access_token, channel['channel_id'],
                    CONFIG['max_messages_per_user'] - total_messages
                )
                if messages:
                    await self.db_manager.store_user_messages(user_id, messages)
                    total_messages += len(messages)
                    logger.info(f"Indexed {len(messages)} messages from {channel['channel_name']}")
                await asyncio.sleep(0.5)

            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    UPDATE authorized_users
                    SET indexing_status = 'completed', last_indexed_at = ?, total_messages_indexed = ?
                    WHERE user_id = ?
                """, (datetime.now().isoformat(), total_messages, user_id))
                await conn.commit()

            logger.info(f"Completed indexing {total_messages} messages for user {user_id}")
            await self._notify_user(user_id, total_messages)

        except Exception as e:
            logger.error(f"Error indexing user {user_id}: {e}")
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    "UPDATE authorized_users SET indexing_status = 'error' WHERE user_id = ?",
                    (user_id,)
                )
                await conn.commit()

    async def _get_user_channels(self, access_token: str, user_id: str) -> List[Dict]:
        channels = []
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {access_token}'}) as session:
            try:
                async with session.get(f"{DISCORD_API_BASE}/users/@me/channels") as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        for channel in json.loads(body):
                            if channel['type'] == 1:
                                name = channel.get('recipients', [{}])[0].get('username', 'DM')
                            elif channel['type'] == 3:
                                name = channel.get('name', 'Group DM')
                            else:
                                continue
                            channels.append({
                                'channel_id': channel['id'], 'channel_name': name,
                                'channel_type': 'dm' if channel['type'] == 1 else 'group_dm'
                            })
                    else:
                        logger.error(f"[{user_id}] /users/@me/channels -> {resp.status}: {body[:300]}")

                async with session.get(f"{DISCORD_API_BASE}/users/@me/guilds") as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        guilds = json.loads(body)
                        logger.info(f"[{user_id}] found {len(guilds)} guilds")
                        for guild in guilds:
                            async with session.get(
                                f"{DISCORD_API_BASE}/guilds/{guild['id']}/channels"
                            ) as resp2:
                                body2 = await resp2.text()
                                if resp2.status == 200:
                                    for channel in json.loads(body2):
                                        if channel['type'] in [0, 5]:
                                            channels.append({
                                                'channel_id': channel['id'],
                                                'channel_name': f"#{channel['name']} ({guild['name']})",
                                                'channel_type': 'guild'
                                            })
                                else:
                                    # This is the expected failure point: a user OAuth2
                                    # token (scopes identify/guilds/messages.read) cannot
                                    # list a guild's channels - that requires bot-level
                                    # permissions in that guild, not a user token.
                                    logger.error(
                                        f"[{user_id}] /guilds/{guild['id']}/channels -> "
                                        f"{resp2.status}: {body2[:300]}"
                                    )
                    else:
                        logger.error(f"[{user_id}] /users/@me/guilds -> {resp.status}: {body[:300]}")
            except Exception as e:
                logger.error(f"Error getting channels for user {user_id}: {e}")
        logger.info(f"[{user_id}] total channels resolved: {len(channels)}")
        return channels

    async def _fetch_channel_messages(self, access_token: str, channel_id: str, limit: int = 100) -> List[DiscordMessage]:
        messages = []
        before = None
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {access_token}'}) as session:
            while len(messages) < limit:
                params = {'limit': min(100, limit - len(messages))}
                if before:
                    params['before'] = before
                async with session.get(
                    f"{DISCORD_API_BASE}/channels/{channel_id}/messages", params=params
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data:
                            break
                        for msg_data in data:
                            messages.append(self._parse_message(msg_data, channel_id))
                        before = data[-1]['id']
                        if len(data) == 100:
                            await asyncio.sleep(0.5)
                    elif resp.status == 429:
                        retry_after = (await resp.json()).get('retry_after', 5)
                        await asyncio.sleep(retry_after)
                    else:
                        logger.error(f"Error fetching messages: {resp.status}")
                        break
        return messages

    def _parse_message(self, msg_data: Dict, channel_id: str) -> DiscordMessage:
        return DiscordMessage(
            id=msg_data['id'], channel_id=channel_id,
            author_id=msg_data['author']['id'],
            author_name=msg_data['author'].get('global_name') or msg_data['author']['username'],
            content=msg_data.get('content', ''),
            timestamp=datetime.fromisoformat(msg_data['timestamp'].replace('Z', '+00:00')),
            edited_timestamp=datetime.fromisoformat(msg_data['edited_timestamp'].replace('Z', '+00:00'))
                if msg_data.get('edited_timestamp') else None,
            attachments=msg_data.get('attachments', []),
            embeds=msg_data.get('embeds', []),
            mentions=[m['id'] for m in msg_data.get('mentions', [])],
            reactions=[{'emoji': r['emoji'], 'count': r['count']} for r in msg_data.get('reactions', [])],
            referenced_message_id=msg_data.get('referenced_message', {}).get('id'),
            is_bot=msg_data['author'].get('bot', False)
        )

    async def _notify_user(self, user_id: str, total_messages: int):
        try:
            user = await discord_client.fetch_user(int(user_id))
            message = (
                f"✅ **Indexing Complete!**\n\n"
                f"I've indexed {total_messages} messages from your Discord history.\n\n"
                f"🌐 Visit the web interface to search:\n"
                f"{CONFIG['web_base_url']}/dashboard?user={user_id}\n\n"
                f"Commands:\n"
                f"• `!status` - Check status\n"
                f"• `!revoke` - Revoke access"
            )
            await user.send(message)
        except Exception as e:
            logger.error(f"Error notifying user {user_id}: {e}")


# ---------------------------------------------------------------------------
# Discord Gateway client (this is what makes the bot appear Online + lets it
# react to DMs instantly via events instead of REST polling)
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.dm_messages = True
intents.message_content = True  # must also be enabled in the Dev Portal

discord_client = discord.Client(intents=intents)


@discord_client.event
async def on_ready():
    logger.info(f"Bot connected as {discord_client.user} (Gateway online)")
    await discord_client.change_presence(activity=discord.Game(name="Indexing your memories"))


@discord_client.event
async def on_message(message: discord.Message):
    if message.author == discord_client.user:
        return
    if not isinstance(message.channel, discord.DMChannel):
        return  # only handle DMs

    content = message.content.lower().strip()
    user_id = str(message.author.id)
    username = message.author.name

    if content.startswith('!authorize') or content.startswith('!auth'):
        await handle_authorize(message.channel, user_id, username)
    elif content.startswith('!status'):
        await handle_status(message.channel, user_id)
    elif content.startswith('!revoke'):
        await handle_revoke(message.channel, user_id)
    elif content.startswith('!help'):
        await send_help(message.channel)


async def handle_authorize(channel, user_id: str, username: str):
    state = secrets.token_urlsafe(32)
    async with db_manager.get_connection() as conn:
        await conn.execute("""
            INSERT INTO oauth_states (state, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (state, user_id, datetime.now().isoformat(),
              (datetime.now() + timedelta(minutes=10)).isoformat()))
        await conn.commit()

    auth_url = (
        f"{DISCORD_OAUTH_BASE}?"
        f"client_id={CONFIG['client_id']}&"
        f"redirect_uri={CONFIG['redirect_uri']}&"
        f"response_type=code&"
        f"scope=identify%20guilds%20messages.read&"
        f"state={state}"
    )

    message = (
        f"🔐 **Discord Recall AI - Authorization**\n\n"
        f"Click the link below to authorize me to access your Discord history:\n"
        f"{auth_url}\n\n"
        f"**What I'll access:**\n"
        f"• Your messages from DMs and servers\n"
        f"• Your guilds and channels\n"
        f"• Your user profile (username, avatar)\n\n"
        f"⚠️ This link expires in 10 minutes.\n"
        f"⚠️ You can revoke access at any time.\n\n"
        f"Questions? Use `!help` for more info."
    )
    await channel.send(message)
    logger.info(f"Sent OAuth2 link to user {username} ({user_id})")


async def handle_status(channel, user_id: str):
    async with db_manager.get_connection() as conn:
        cursor = await conn.execute("SELECT * FROM authorized_users WHERE user_id = ?", (user_id,))
        user = await cursor.fetchone()

    if user:
        status = user['indexing_status']
        total = user['total_messages_indexed'] or 0
        last_indexed = user['last_indexed_at'] or "Never"
        message = (
            f"📊 **Your Archive Status**\n\n"
            f"Status: {status}\n"
            f"Messages Indexed: {total}\n"
            f"Last Indexed: {last_indexed}\n\n"
            f"Commands:\n"
            f"• `!status` - Check status\n"
            f"• `!revoke` - Revoke access\n"
            f"• `!help` - Help"
        )
    else:
        message = "You haven't authorized me yet. Use `!authorize` to start."
    await channel.send(message)


async def handle_revoke(channel, user_id: str):
    async with db_manager.get_connection() as conn:
        await conn.execute("UPDATE authorized_users SET is_active = 0 WHERE user_id = ?", (user_id,))
        await conn.commit()
    await channel.send("🔒 Access revoked. Your data has been removed from my system.")
    logger.info(f"Revoked access for user {user_id}")


async def send_help(channel):
    message = (
        "🤖 **Discord Recall AI - Commands**\n\n"
        "`!authorize` - Start OAuth2 authorization\n"
        "`!status` - Check your archive status\n"
        "`!revoke` - Revoke my access\n"
        "`!help` - Show this message\n\n"
        "After authorizing, visit the web interface to search your messages."
    )
    await channel.send(message)


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return HTMLResponse("""
    <!DOCTYPE html><html><head><title>Discord Recall AI</title>
    <style>
        body{font-family:'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;
             min-height:100vh;display:flex;justify-content:center;align-items:center}
        .container{max-width:700px;padding:40px;text-align:center}
        h1{font-size:3em;background:linear-gradient(135deg,#5865F2,#404EED);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .card{background:#1a1a1a;padding:30px;border-radius:12px;border:1px solid #2a2a2a;margin-top:20px}
        code{background:#2a2a2a;padding:2px 8px;border-radius:4px;color:#5865F2}
    </style></head><body><div class="container">
        <h1>📚 Discord Recall AI</h1>
        <div class="card"><h2>🔐 OAuth2 Authorization</h2>
        <p>DM the bot <code>!authorize</code> to begin.</p></div>
    </div></body></html>
    """)


@app.get("/oauth2/callback")
async def oauth2_callback(request: Request):
    params = dict(request.query_params)
    code = params.get('code')
    state = params.get('state')
    error = params.get('error')

    if error:
        return HTMLResponse(f"<h1>Authorization Error</h1><p>{error}</p>")
    if not code or not state:
        return HTMLResponse("<h1>Invalid request</h1><p>Missing code or state</p>")

    async with db_manager.get_connection() as conn:
        cursor = await conn.execute("""
            SELECT * FROM oauth_states
            WHERE state = ? AND used = 0 AND expires_at > datetime('now')
        """, (state,))
        state_data = await cursor.fetchone()
        if not state_data:
            return HTMLResponse("<h1>Invalid or expired state</h1><p>Please try again with !authorize</p>")
        user_id = state_data['user_id']
        await conn.execute("UPDATE oauth_states SET used = 1 WHERE state = ?", (state,))
        await conn.commit()

    async with aiohttp.ClientSession() as session:
        data = {
            'client_id': CONFIG['client_id'], 'client_secret': CONFIG['client_secret'],
            'grant_type': 'authorization_code', 'code': code,
            'redirect_uri': CONFIG['redirect_uri']
        }
        async with session.post(DISCORD_TOKEN_URL, data=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise HTTPException(status_code=400, detail=f"Token exchange failed: {error_text}")
            token_data = await resp.json()
            access_token = token_data['access_token']
            refresh_token = token_data.get('refresh_token')
            expires_in = token_data['expires_in']

        async with session.get(DISCORD_USER_URL, headers={'Authorization': f'Bearer {access_token}'}) as user_resp:
            if user_resp.status != 200:
                raise HTTPException(status_code=400, detail="Failed to get user info")
            user_data = await user_resp.json()
            username = user_data['username']

    await db_manager.store_oauth_token(user_id, username, access_token, refresh_token, expires_in)
    asyncio.create_task(indexer.index_user(user_id, access_token))

    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><title>Authorization Successful</title>
    <style>
        body{{font-family:'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;
             display:flex;justify-content:center;align-items:center;height:100vh;text-align:center}}
        .container{{background:#1a1a1a;padding:40px;border-radius:12px;border:1px solid #2a2a2a;max-width:500px}}
        h1{{color:#2ecc71}}
        .btn{{display:inline-block;padding:12px 30px;background:#5865F2;color:#fff;
             text-decoration:none;border-radius:8px;margin-top:20px}}
    </style></head><body><div class="container">
        <h1>✅ Authorization Successful!</h1>
        <p>Hi <strong>{username}</strong>!</p>
        <p>I'm now indexing your Discord messages. You'll get a DM when it's done.</p>
        <a href="/dashboard?user={user_id}" class="btn">Go to Dashboard</a>
    </div></body></html>
    """)


@app.get("/dashboard")
async def dashboard(user: str = Query(...)):
    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><title>Discord Recall AI - Dashboard</title>
    <style>
        body{{font-family:'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}}
        .container{{max-width:1000px;margin:0 auto}}
        .header{{background:linear-gradient(135deg,#5865F2,#404EED);padding:20px 30px;
                border-radius:12px;margin-bottom:20px;display:flex;justify-content:space-between}}
        .search-box{{background:#1a1a1a;padding:20px;border-radius:12px;border:1px solid #2a2a2a;margin-bottom:20px}}
        .search-box input{{width:100%;padding:12px;background:#2a2a2a;border:2px solid #3a3a3a;
                           border-radius:8px;color:#e0e0e0;font-size:16px}}
        .search-box button{{margin-top:10px;padding:12px 30px;background:#5865F2;color:#fff;
                            border:none;border-radius:8px;font-size:16px;cursor:pointer}}
        .results{{background:#1a1a1a;border-radius:12px;border:1px solid #2a2a2a;padding:20px;min-height:200px}}
        .result-item{{padding:15px;border-bottom:1px solid #2a2a2a}}
        .result-author{{color:#5865F2;font-weight:600}}
        .loading{{text-align:center;padding:40px;color:#808080}}
    </style></head><body><div class="container">
        <div class="header"><h1>📚 Discord Recall AI</h1><span>User: {user[:8]}...</span></div>
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search your messages..." />
            <button onclick="search()">🔍 Search</button>
        </div>
        <div class="results" id="results"><div class="loading">Enter a search query to begin</div></div>
    </div>
    <script>
        const userId = '{user}';
        async function search() {{
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;
            document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';
            try {{
                const response = await fetch(`/api/search?user=${{userId}}&q=${{encodeURIComponent(query)}}`);
                const data = await response.json();
                if (data.results && data.results.length > 0) {{
                    document.getElementById('results').innerHTML = data.results.map(item => `
                        <div class="result-item">
                            <div><span class="result-author">${{escapeHtml(item.author_name)}}</span>
                            <span> ${{new Date(item.timestamp).toLocaleString()}}</span></div>
                            <div>${{escapeHtml(item.content)}}</div>
                        </div>`).join('');
                }} else {{
                    document.getElementById('results').innerHTML = '<div class="loading">No results found</div>';
                }}
            }} catch (e) {{
                document.getElementById('results').innerHTML = '<div class="loading">Error searching</div>';
            }}
        }}
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }}
        document.getElementById('searchInput').addEventListener('keypress', e => {{ if (e.key==='Enter') search(); }});
    </script></body></html>
    """)


@app.get("/api/search")
async def api_search(user: str, q: str, limit: int = 50):
    try:
        results = await db_manager.search_user_messages(user, q, limit)
        return {'results': results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Globals initialized in main()
db_manager: Optional[DatabaseManager] = None
embedding_service: Optional[EmbeddingService] = None
indexer: Optional[UserIndexer] = None


async def main():
    global db_manager, embedding_service, indexer

    logger.info("Starting Discord Recall AI - discord.py Edition...")

    embedding_service = EmbeddingService(CONFIG['model_cache_dir'])
    embedding_service.initialize()

    db_manager = DatabaseManager(CONFIG['database_path'])
    indexer = UserIndexer(db_manager, embedding_service)

    config = uvicorn.Config(app, host="0.0.0.0", port=CONFIG['web_port'], log_level="info")
    web_server = uvicorn.Server(config)

    # Run the web server and the Discord Gateway client concurrently,
    # in the SAME event loop. This is the key fix: previously the bot
    # never opened a Gateway/WebSocket connection at all, so Discord
    # always showed it as offline even though the REST/HTTP side worked.
    async with discord_client:
        await asyncio.gather(
            discord_client.start(CONFIG['bot_token']),
            web_server.serve(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")