import os
import asyncio
import sys
import json
import re as _re
import aiohttp
import discord
from discord.ext import commands
import groq
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HIBP_API_KEY = os.getenv("HIBP_API_KEY", "")

groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


async def groq_request(messages, max_tokens=512):
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY not set")
    try:
        return await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            ).choices[0].message.content
        )
    except Exception as e:
        return f"ERROR: {e}"


async def sherlock_search(username):
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "sherlock_project",
            username, "--print-found", "--timeout", "10",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.abspath("sherlock"),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        lines = stdout.decode(errors="ignore").splitlines()
        return [l.strip() for l in lines if "[+]" in l or ("http" in l.lower() and "[+]" in l)]
    except Exception:
        return []


async def maigret_search(username):
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "maigret",
            username, "--print-found", "--no-color",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.abspath("maigret"),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
        lines = stdout.decode(errors="ignore").splitlines()
        return [_re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in lines if "[+]" in l and "http" in l.lower()]
    except Exception:
        return []


async def holehe_search(email):
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "holehe", email, "--only-used", "--no-color",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
        lines = stdout.decode(errors="ignore").splitlines()
        return [_re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in lines if l.strip()]
    except Exception:
        return []


async def github_search(target, search_type="users"):
    results = []
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "DimaOSINT/1.0"}
    urls = {
        "users": f"https://api.github.com/search/users?q={target}&per_page=5",
        "repos": f"https://api.github.com/search/repositories?q={target}&per_page=5",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(urls.get(search_type, urls["users"]), headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for it in data.get("items", [])[:5]:
                        results.append(it.get("html_url", ""))
    except Exception:
        pass
    return results


async def ip_lookup(ip):
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://ip-api.com/json/{ip}?fields=status,message,country,regionName,city,zip,lat,lon,timezone,isp,org,as,proxy,hosting,query"
            async with session.get(url, timeout=10) as resp:
                return await resp.json()
    except Exception:
        return {}


async def crtsh_search(domain):
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://crt.sh/?q=%.{domain}&output=json"
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    seen = set()
                    out = []
                    for cert in data[:50]:
                        name = cert.get("name_value", "")
                        for sub in name.split("\n"):
                            sub = sub.strip()
                            if sub and not sub.startswith("*") and sub not in seen:
                                seen.add(sub)
                                out.append(sub)
                    return out
    except Exception:
        pass
    return []


async def duckduckgo_search(queries):
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for query in queries[:5]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        for match in _re.finditer(r'href="//duckduckgo\.com/l/\?uddg=([^&"]+)', html):
                            encoded = match.group(1)
                            try:
                                real_url = _re.sub(r'%([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), encoded)
                                if real_url.startswith('http'):
                                    results.append(real_url)
                            except Exception:
                                pass
        except Exception:
            pass
    return list(dict.fromkeys(results))[:10]


async def phone_lookup(phone):
    results = []
    clean = phone.strip()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.veriphone.io/v2/verify?phone={clean}", timeout=8) as resp:
                if resp.status == 200:
                    d = await resp.json()
                    results.append(str(d))
    except Exception:
        pass
    return results


async def hibp_check(email):
    if not HIBP_API_KEY:
        return []
    out = []
    try:
        headers = {"hibp-api-key": HIBP_API_KEY, "user-agent": "DimaOSINT/1.0"}
        url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{email}?truncateResponse=false"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    for b in await resp.json():
                        out.append(b.get("Name"))
    except Exception:
        pass
    return out


def extract_email_username(email):
    return email.split("@")[0] if "@" in email else email


DETECT_PATTERNS = [
    (r'@\w+', 'osint_username'),
    (r'[\w\.-]+@[\w\.-]+\.\w+', 'osint_email'),
    (r'\+?\d[\d\s\-()]{8,}', 'osint_phone'),
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'osint_ip'),
    (r'[\w-]+\.(com|org|net|io|me|dev|co)', 'osint_domain'),
]


def detect_intent_local(text):
    for pattern, intent_type in DETECT_PATTERNS:
        match = _re.search(pattern, text, _re.IGNORECASE)
        if match:
            target = match.group(0)
            if intent_type == 'osint_username':
                target = target.lstrip('@')
            return {"intent": intent_type, "target": target, "filters": {}}
    return None


def _build_base_queries(target, intent_type, filters):
    q = []
    if intent_type in ("osint_name", "osint_lastname"):
        q += [
            f'site:facebook.com "{target}"',
            f'site:instagram.com "{target}"',
            f'site:linkedin.com "{target}"',
        ]
    elif intent_type == "osint_username":
        q += [f'site:github.com "{target}"', f'"{target}" profile', f'site:twitter.com "{target}"']
    elif intent_type == "osint_email":
        q += [f'"{target}"', f'intext:"{target}"']
    elif intent_type == "osint_phone":
        q += [f'"{target}"']
    elif intent_type == "osint_domain":
        q += [f'site:{target}']
    return [x for x in dict.fromkeys(q) if x]


async def generate_smart_queries(intent):
    target = intent.get("target", "")
    intent_type = intent.get("intent", "chat")
    base = _build_base_queries(target, intent_type, intent.get("filters", {}))
    if len(base) >= 6 or not groq_client:
        return base
    prompt = f'Generate 3 short search queries for {target} (type={intent_type}). Return JSON {{"queries": [...]}}'
    result = await groq_request([{"role": "user", "content": prompt}], max_tokens=128)
    try:
        obj = json.loads(result.strip())
        extra = obj.get("queries", [])
        return list(dict.fromkeys(base + extra))
    except Exception:
        return base


async def gather_all_osint(intent):
    intent_type = intent.get("intent", "chat")
    target = intent.get("target", "")
    queries = await generate_smart_queries(intent)
    res = {}

    if intent_type == "osint_username":
        res["sherlock"] = await sherlock_search(target)
        res["maigret"] = await maigret_search(target)
        res["github"] = await github_search(target)
        res["web"] = await duckduckgo_search(queries)
    elif intent_type == "osint_email":
        res["holehe"] = await holehe_search(target)
        res["hibp"] = await hibp_check(target)
        res["github"] = await github_search(extract_email_username(target))
        res["web"] = await duckduckgo_search(queries)
    elif intent_type == "osint_ip":
        res["ip"] = await ip_lookup(target)
        res["web"] = await duckduckgo_search(queries)
    elif intent_type == "osint_domain":
        res["crtsh"] = await crtsh_search(target)
        res["github"] = await github_search(target, "repos")
        res["web"] = await duckduckgo_search(queries)
    elif intent_type == "osint_phone":
        res["phone"] = await phone_lookup(target)
        res["web"] = await duckduckgo_search(queries)
    elif intent_type in ("osint_name", "osint_lastname"):
        res["github"] = await github_search(target)
        res["web"] = await duckduckgo_search(queries)
    else:
        res["web"] = await duckduckgo_search(queries)

    res["queries_used"] = queries
    return res


INTENT_PROMPT = (
    "Extract intent and target from the user's message.\n"
    "Return ONLY valid JSON: {\"intent\":..., \"target\":..., \"filters\":{}}\n"
    "Valid intents: osint_username, osint_name, osint_lastname, osint_email, osint_phone, osint_domain, osint_ip, chat\n\n"
    "Examples:\n"
    "- 'find info on john doe' -> {\"intent\":\"osint_name\",\"target\":\"john doe\",\"filters\":{}}\n"
    "- 'lookup @john123' -> {\"intent\":\"osint_username\",\"target\":\"john123\",\"filters\":{}}\n"
    "- 'check user@gmail.com' -> {\"intent\":\"osint_email\",\"target\":\"user@gmail.com\",\"filters\":{}}\n"
)


async def parse_intent(text):
    local = detect_intent_local(text)
    if local:
        return local

    if not groq_client:
        return {"intent": "chat", "target": text, "filters": {}, "confidence": 0.5}

    prompt = INTENT_PROMPT + "\nUser: " + text
    result = await groq_request([{"role": "user", "content": prompt}], max_tokens=256)
    try:
        raw = result.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception:
        return {"intent": "chat", "target": text, "filters": {}, "confidence": 0.2}


intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)
    if isinstance(message.channel, discord.DMChannel) and not message.content.startswith('!'):
        await handle_dm_query(message)


@bot.command(name='osint')
async def cmd_osint(ctx, *, query=None):
    if not query:
        await ctx.send("Usage: `!osint <query>`\nExample: `!osint john doe` or `!osint @username`")
        return
    await ctx.defer()
    intent = await parse_intent(query)
    data = await gather_all_osint(intent)
    parts = [f"Intent: {intent.get('intent')} | Target: {intent.get('target')}\n"]
    for k, v in data.items():
        if k == 'queries_used':
            parts.append(f"Queries: {len(v)}")
        else:
            count = len(v) if isinstance(v, list) else (1 if v else 0)
            parts.append(f"{k}: {count}")
    await ctx.send("\n".join(parts))


async def handle_dm_query(message):
    query = message.content.strip()
    if not query:
        return
    async with message.channel.typing():
        intent = await parse_intent(query)
        data = await gather_all_osint(intent)
    parts = [f"Intent: {intent.get('intent')} | Target: {intent.get('target')}\n"]
    for k, v in data.items():
        if k == 'queries_used':
            parts.append(f"Queries: {len(v)}")
        else:
            count = len(v) if isinstance(v, list) else (1 if v else 0)
            parts.append(f"{k}: {count}")
    await message.reply("\n".join(parts))


if __name__ == '__main__':
    if not DISCORD_TOKEN:
        print("Set DISCORD_TOKEN in .env")
        sys.exit(1)
    if not GROQ_API_KEY:
        print("Set GROQ_API_KEY in .env (free at console.groq.com)")
        sys.exit(1)
    bot.run(DISCORD_TOKEN)
