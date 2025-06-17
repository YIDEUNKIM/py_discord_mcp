"""
discord_mcp_server.py — Discord ↔ MCP bridge bot (UTF-8 patch integrated)
------------------------------------------------------------------------
• Forces all I/O streams to UTF-8 so Korean (and any Unicode) text never breaks
• Sets json.dumps default ensure_ascii=False
"""

# ================================================================
# 0. **Global UTF-8 enforcement patch**
#    -------------------------------
#    Standard streams, Windows console, and JSON serialization are
#    coerced to UTF-8 to avoid mojibake anywhere in the pipeline.
# ================================================================
import sys, os

def _force_utf8():
    """
    Force Python runtime, stdio, Windows console, and json to UTF-8.
    Executed once at import-time before anything else.
    """
    import json as _json

    # 1) Python runtime environment flags
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # 2) Reconfigure stdout / stderr (Python ≥3.7)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    # 3) Windows console code page → UTF-8 (65001)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            # Non-Windows or headless container — ignore
            pass

    # 4) Patch json.dumps so ensure_ascii defaults to False
    _orig_dumps = _json.dumps
    def _dumps_no_ascii(*args, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        return _orig_dumps(*args, **kwargs)
    _json.dumps = _dumps_no_ascii

_force_utf8()

# ================================================================
# 1. Required library imports
# ================================================================
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from functools import wraps
from itertools import combinations
from collections import defaultdict
import json

import discord
from discord.ext import commands
from mcp.server import Server
from mcp.types import Tool, TextContent, EmptyResult
from mcp.server.stdio import stdio_server

# ================================================================
# 2. Initialization · logging · Discord/MCP objects
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]   # stdout is already UTF-8
)
logger = logging.getLogger("discord_mcp_server")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True
bot = commands.Bot(command_prefix="!", intents=intents)

app = Server("discord_server")
discord_client: Optional[commands.Bot] = None

# ================================================================
# 3. File persistence · constants · channel categorisation helpers
# ================================================================
LAST_SEEN_FILE        = "last_seen.json"
INACTIVE_DAYS         = 7
SUMMARY_MAX_CHARS     = 1000

NOTICE_LIMIT, CAL_LIMIT, TALK_LIMIT = 30, 30, 300

NOTICE_CH   = {"rules", "rule", "notice", "announcements"}
CALENDAR_CH = {"calendar", "calender", "schedule"}
TALK_CH     = {"general", "chat", "talk", "discussion"}

last_seen: Dict[str, str] = {}

def _load_last_seen():
    """Load persisted last-seen timestamps from disk (if any)."""
    global last_seen
    if os.path.exists(LAST_SEEN_FILE):
        try:
            with open(LAST_SEEN_FILE, encoding="utf-8") as f:
                last_seen = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {LAST_SEEN_FILE}: {e}")

def _save_last_seen():
    """Persist current last-seen state to disk."""
    try:
        with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(last_seen, f, indent=2)   # ensure_ascii=False already patched
    except Exception as e:
        logger.warning(f"Failed to save {LAST_SEEN_FILE}: {e}")

# ================================================================
# 4. Channel categorisation · message collection
# ================================================================
async def _categorize(ch: discord.TextChannel) -> Optional[str]:
    """Return 'notice' | 'calendar' | 'talk' or None for uncategorised channels."""
    n = ch.name.lower()
    if n in NOTICE_CH:   return "notice"
    if n in CALENDAR_CH: return "calendar"
    if n in TALK_CH:     return "talk"
    return None

async def _gather_messages(guild: discord.Guild) -> Dict[str, List[str]]:
    """
    Collect recent messages from each categorised channel in the guild.
    Messages are returned oldest → newest for easier summarisation.
    """
    buckets = {"notice": [], "calendar": [], "talk": []}
    for ch in guild.text_channels:
        kind = await _categorize(ch)
        if not kind:
            continue
        limit = {"notice": NOTICE_LIMIT, "calendar": CAL_LIMIT, "talk": TALK_LIMIT}[kind]
        try:
            msgs = []
            async for msg in ch.history(limit=limit):
                msgs.append(f"{msg.author.display_name}: {msg.content}")
            buckets[kind].extend(reversed(msgs))   # chronological order
        except Exception as e:
            logger.warning(f"History fetch failed for channel {ch.name}: {e}")
    return buckets

# ================================================================
# 5. Discord events · guard decorator
# ================================================================
@bot.event
async def on_ready():
    """Called once the Discord connection is fully established."""
    global discord_client
    discord_client = bot
    logger.info(f"Logged in as {bot.user.name}")
    _load_last_seen()

def require_discord_client(func):
    """Prevent tool calls before the Discord client is ready."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("Discord client not ready")
        return await func(*args, **kwargs)
    return wrapper

# ================================================================
# 6. MCP tool definitions & dispatcher
# ================================================================
@app.list_tools()
async def list_tools() -> List[Tool]:
    """Expose available tools to the MCP runtime."""
    return [
        Tool(
            name="get_server_info",
            description="Get information about a Discord server",
            inputSchema={
                "type": "object",
                "properties": {"server_id": {"type": "string"}},
                "required": ["server_id"]
            }
        ),
        Tool(
            name="role_analysis",
            description="Analyse user role in a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"},
                    "limit": {"type": "number", "minimum": 1, "maximum": 100}
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="get_user_info",
            description="Get information about a Discord user",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"]
            }
        ),
        Tool(
            name="list_members",
            description="Get a list of members in a server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "limit": {"type": "number", "minimum": 1, "maximum": 1000}
                },
                "required": ["server_id"]
            }
        ),
        Tool(
            name="list_channels",
            description="List text channels in a server",
            inputSchema={
                "type": "object",
                "properties": {"server_id": {"type": "string"}},
                "required": ["server_id"]
            }
        ),
        Tool(
            name="relationship_analysis",
            description="Analyse relationships between users in a channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"},
                    "limit": {"type": "number", "minimum": 1, "maximum": 100}
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="summarize_text",
            description="Collect recent messages (notice/calendar/chat) and return raw text",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"]
            }
        )
    ]

@app.call_tool()
@require_discord_client
async def call_tools(name: str, arguments: Any) -> List[TextContent]:
    """Main dispatcher for all tool calls."""
    # ---------- get_server_info ----------
    if name == "get_server_info":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        info = {
            "name": guild.name,
            "id": str(guild.id),
            "owner_id": str(guild.owner_id),
            "member_count": guild.member_count,
            "created_at": guild.created_at.isoformat(),
            "description": guild.description,
            "premium_tier": guild.premium_tier,
            "explicit_content_filter": str(guild.explicit_content_filter)
        }
        return [TextContent(
            type="text",
            text="Server Information:\n" +
                 "\n".join(f"{k}: {v}" for k, v in info.items())
        )]

    # ---------- role_analysis ----------
    elif name == "role_analysis":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        limit = min(int(arguments.get("limit", 10)), 100)
        messages = []

        async for message in channel.history(limit=limit):
            reactions = [{
                "emoji": str(r.emoji.name or r.emoji.id or r.emoji),
                "count": r.count
            } for r in message.reactions]

            messages.append({
                "id": str(message.id),
                "author": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "reactions": reactions
            })

        fmt = lambda r: f"{r['emoji']}({r['count']})"
        return [TextContent(
            type="text",
            text=f"Retrieved {len(messages)} messages:\n\n" + "\n".join(
                f"{m['author']} ({m['timestamp']}): {m['content']}\n"
                f"Reactions: {', '.join(fmt(r) for r in m['reactions']) or 'No reactions'}"
                for m in messages)
        )]

    # ---------- get_user_info ----------
    elif name == "get_user_info":
        user = await discord_client.fetch_user(int(arguments["user_id"]))
        info = {
            "id": str(user.id),
            "name": user.name,
            "discriminator": user.discriminator,
            "bot": user.bot,
            "created_at": user.created_at.isoformat()
        }
        return [TextContent(
            type="text",
            text=(
                "User information:\n"
                f"Name: {info['name']}#{info['discriminator']}\n"
                f"ID: {info['id']}\nBot: {info['bot']}\n"
                f"Created: {info['created_at']}"
            )
        )]

    # ---------- relationship_analysis ----------
    elif name == "relationship_analysis":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        limit = min(int(arguments.get("limit", 10)), 100)
        fetch_users = arguments.get("fetch_users", False)

        messages = []
        async for msg in channel.history(limit=limit):
            reactions = []
            for r in msg.reactions:
                users = []
                if fetch_users:   # optionally fetch usernames of reactors
                    try:
                        users = [f"{u.name}#{u.discriminator}" for u in await r.users().flatten()]
                    except Exception as e:
                        logger.warning(f"Failed to fetch reaction users: {e}")
                reactions.append({
                    "emoji": str(r.emoji.name or r.emoji.id or r.emoji),
                    "count": r.count,
                    "users": users
                })

            messages.append({
                "author": str(msg.author),
                "author_id": str(msg.author.id),
                "content": msg.content,
                "mentions": [str(u.id) for u in msg.mentions],
                "timestamp": msg.created_at.isoformat(),
                "is_reply": bool(msg.reference),
                "reactions": reactions
            })

        messages.sort(key=lambda m: m["timestamp"])

        # Text summary of all messages
        summary = "Message Summary:\n\n"
        fmt = lambda r: f"{r['emoji']}({r['count']})" + (f": {', '.join(r['users'])}" if r["users"] else "")
        for m in messages:
            summary += (
                f"{m['author']} ({m['timestamp']}): {m['content']}\n"
                f"Reactions: {', '.join(fmt(r) for r in m['reactions']) or 'No reactions'}\n\n"
            )

        # Compute pairwise interaction statistics
        user_ids = {m["author_id"]: m["author"] for m in messages}
        stats = defaultdict(lambda: {"mentions": 0, "replies": 0, "reactions": 0})

        for m in messages:
            sender = m["author"]
            for mid in m["mentions"]:
                tgt = user_ids.get(mid)
                if tgt and tgt != sender:
                    stats[tuple(sorted([sender, tgt]))]["mentions"] += 1
            if m["is_reply"]:
                for tgt in user_ids.values():
                    if tgt != sender:
                        stats[tuple(sorted([sender, tgt]))]["replies"] += 1
            for tgt in user_ids.values():
                if tgt != sender:
                    stats[tuple(sorted([sender, tgt]))]["reactions"] += sum(r["count"] for r in m["reactions"])

        def describe(pair, s):
            total = sum(s.values())
            if not total:
                return f"{pair[0]} <-> {pair[1]}: no significant interaction."
            kinds = [k for k, v in s.items() if v]
            return f"{pair[0]} <-> {pair[1]}: interaction via {', '.join(kinds)}."

        relationships = "\n".join(describe(p, s) for p, s in stats.items())

        report = (
            "Relationship Analysis Report\n"
            "=============================\n\n"
            f"Total messages analysed: {len(messages)}\n\n"
            f"{summary}\n{relationships}"
        )
        return [TextContent(type="text", text=report)]

    # ---------- summarize_text ----------
    elif name == "summarize_text":
        uid = arguments["user_id"]
        try:
            user_id = int(uid)
        except ValueError:
            return [TextContent(type="text", text="Invalid user_id format")]

        guild = next((g for g in discord_client.guilds if g.get_member(user_id)), None)
        if not guild:
            return [TextContent(type="text", text="User not in a mutual guild.")]

        buckets = await _gather_messages(guild)
        parts = []
        if buckets["notice"]:   parts.append("【Rules/Notice】\n"  + "\n".join(buckets["notice"]))
        if buckets["calendar"]: parts.append("【Calendar】\n"     + "\n".join(buckets["calendar"]))
        if buckets["talk"]:     parts.append("【General/Chat】\n" + "\n".join(buckets["talk"]))

        return [TextContent(type="text", text="\n\n".join(parts) or "No new messages.")]

    # ---------- list_members ----------
    elif name == "list_members":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        limit = min(int(arguments.get("limit", 100)), 1000)
        members = []
        async for m in guild.fetch_members(limit=limit):
            members.append({
                "id": str(m.id),
                "name": m.name,
                "nick": m.nick,
                "joined_at": m.joined_at.isoformat() if m.joined_at else None,
                "roles": [str(r.id) for r in m.roles[1:]]   # skip @everyone
            })

        return [TextContent(
            type="text",
            text=f"Server Members ({len(members)}):\n" + "\n".join(
                f"{m['name']} (ID: {m['id']}, Roles: {', '.join(m['roles'])})" for m in members)
        )]

    # ---------- list_channels ----------
    elif name == "list_channels":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        chans = [
            ch for ch in discord_client.get_all_channels()
            if isinstance(ch, discord.TextChannel) and ch.guild.id == guild.id
        ]
        return [TextContent(
            type="text",
            text="Text Channels:\n" + "\n".join(f"{ch.name} (ID: {ch.id})" for ch in chans)
        )]

    # ---------- unknown ----------
    return []

# ================================================================
# 7. Discord event: member status change
# ================================================================
@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """
    When a member comes online after INACTIVE_DAYS, trigger the
    summarize_text tool so they catch up on what they missed.
    """
    if before.status == discord.Status.offline and after.status != discord.Status.offline:
        uid = str(after.id)
        now = datetime.utcnow()

        last_dt = None
        if uid in last_seen:
            try:
                last_dt = datetime.fromisoformat(last_seen[uid])
            except ValueError:
                logger.warning(f"Bad ISO date for {uid}: {last_seen[uid]}")

        if not last_dt or (now - last_dt).days >= INACTIVE_DAYS:
            logger.info(f"{after.display_name} online after {INACTIVE_DAYS}+ days → summarising")
            try:
                await app.request_tool("summarize_text", {"user_id": uid})
            except Exception as e:
                logger.error(f"summarize_text error for {uid}: {e}")

        last_seen[uid] = now.isoformat()
        _save_last_seen()

# ================================================================
# 8. Main entry-point
# ================================================================
async def main():
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
