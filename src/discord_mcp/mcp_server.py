"""
import external library
"""
import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from functools import wraps

from itertools import combinations
from collections import defaultdict

import discord
from discord.ext import commands
from mcp.server import Server
from mcp.types import Tool, TextContent, EmptyResult
from mcp.server.stdio import stdio_server
import json

##################################################
"""
Initial Constructor
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_mcp_server")

# Discord bot setup
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True
bot = commands.Bot(command_prefix="!", intents=intents)

app = Server("discord_server")
discord_client: Optional[commands.Bot] = None
##################################################

# ─────────────────── CONSTANTS & FILE HELPERS ───────────────────
LAST_SEEN_FILE        = "last_seen.json"
INACTIVE_DAYS         = 7
SUMMARY_MAX_CHARS     = 1000
NOTICE_LIMIT, CAL_LIMIT, TALK_LIMIT = 30, 30, 300

NOTICE_CH   = {"rules", "rule", "notice", "announcements"}
CALENDAR_CH = {"calendar", "calender", "schedule"}
TALK_CH     = {"general", "chat", "talk", "discussion"}

last_seen: Dict[str, str] = {}

def _load_last_seen():
    """Load last seen data from JSON file."""
    global last_seen
    if os.path.exists(LAST_SEEN_FILE):
        try:
            with open(LAST_SEEN_FILE, encoding="utf-8") as f:
                last_seen = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {LAST_SEEN_FILE}: {e}")

def _save_last_seen():
    """Save last seen data to JSON file."""
    try:
        with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(last_seen, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save {LAST_SEEN_FILE}: {e}")

# ─────────────────── LLM WRAPPER ───────────────────
async def _llm_summarize(text: str) -> str:
    """
    Call the external LLM summary tool (e.g., "llm_summarize") to get ≤1000 characters summary.
    If timeout or error, fallback to the first 1000 characters.
    """
    try:
        # Requests summary from an external tool connected to the MCP server
        res = await asyncio.wait_for(
            app.request_tool("llm_summarize", {"text": text, "max_chars": SUMMARY_MAX_CHARS}),
            timeout=25,
        )
        if res:
            first = res[0]
             # Claude → TextContent
            if isinstance(first, TextContent) and first.type == "text":
                return first.text[:SUMMARY_MAX_CHARS]
            # Claude → raw string (fallback)
            if isinstance(first, str):
                return first[:SUMMARY_MAX_CHARS]
    except Exception as e:
        logger.warning(f"LLM summarize error: {e}")
    return text[:SUMMARY_MAX_CHARS]

# ─────────────────── MESSAGE GATHERING ───────────────────
async def _categorize(ch: discord.TextChannel) -> Optional[str]:
    """Categorize channel by its name."""
    n = ch.name.lower()
    if n in NOTICE_CH:   return "notice"
    if n in CALENDAR_CH: return "calendar"
    if n in TALK_CH:     return "talk"
    return None

async def _gather_messages(guild: discord.Guild) -> Dict[str, List[str]]:
    """
    Gather latest messages from categorized channels in the guild.
    Returns dictionary: {category: [messages]}
    """
    buckets = {"notice": [], "calendar": [], "talk": []}
    for ch in guild.text_channels:
        kind = await _categorize(ch)
        if not kind: continue
        limit = {"notice": NOTICE_LIMIT, "calendar": CAL_LIMIT, "talk": TALK_LIMIT}[kind]
        try:
            channel_messages = []
            async for msg in ch.history(limit=limit):
                channel_messages.append(f"{msg.author.display_name}: {msg.content}")
            buckets[kind].extend(reversed(channel_messages))  # Order: oldest first
        except Exception as e:
            logger.warning(f"History fetch fail for channel {ch.name}: {e}")
    return buckets
##################################################

@bot.event
async def on_ready():
    """Fires when the Discord bot is ready."""
    global discord_client
    discord_client = bot
    logger.info(f"Log as {bot.user.name}")
    _load_last_seen()  # Load last seen data at startup

def require_discord_client(func):
    """Decorator to ensure the Discord client is ready before running the function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("client not ready")
        return await func(*args, **kwargs)
    return wrapper

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Register all tools available via the MCP server."""
    original_tools = [
        Tool(
            name="get_server_info",
            description="Get information about a Discord server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server (guild) ID"
                    }
                },
                "required": ["server_id"]
            }
        ),
        Tool(
            name="role_analysis",
            description="Analyze user role in discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to fetch (max 100)",
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="get_user_info",
            description="Get information about a Discord user",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Discord user ID"
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="relationship_analysis",
            description="Analyze relationships between users in a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to analyze (max 100)",
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="llm_summarize",       # ★ new tool
            description="Return ≤1000-char summary using Claude",
            inputSchema={
                "type": "object",
                " properties": {
                 "text": {"type": "string"},
                 "max_chars": {"type": "number"},
                },
                "required": ["text"],
            }
        ),
    ]
    summarize_text_tool = Tool(
        name="summarize_text",
        description="(Internal) Send ≤1000-char DM summary to returning user",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    )
    original_tools.append(summarize_text_tool)
    return original_tools

@app.call_tool()
@require_discord_client
async def call_tools(name: str, arguments: Any) -> List[TextContent]:
    """Main tool dispatcher called via MCP."""
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
            text=f"Server Information:\n" + "\n".join(f"{k}: {v}" for k, v in info.items())
        )]

    elif name == "role_analysis":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        limit = min(int(arguments.get("limit", 10)), 100)

        messages = []
        async for message in channel.history(limit=limit):
            reaction_data = []
            for reaction in message.reactions:
                emoji_str = str(reaction.emoji.name) if hasattr(reaction.emoji,
                                                                'name') and reaction.emoji.name else str(
                    reaction.emoji.id) if hasattr(reaction.emoji, 'id') else str(reaction.emoji)
                reaction_info = {
                    "emoji": emoji_str,
                    "count": reaction.count
                }
                reaction_data.append(reaction_info)
            messages.append({
                "id": str(message.id),
                "author": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "reactions": reaction_data
            })
        def format_reaction(r):
            return f"{r['emoji']}({r['count']})"
        return [TextContent(
            type="text",
            text=f"Retrieved {len(messages)} messages:\n\n" +
                 "\n".join([
                     f"{m['author']} ({m['timestamp']}): {m['content']}\n" +
                     f"Reactions: {', '.join([format_reaction(r) for r in m['reactions']]) if m['reactions'] else 'No reactions'}"
                     for m in messages
                 ])
        )]

    elif name == "get_user_info":
        user = await discord_client.fetch_user(int(arguments["user_id"]))
        user_info = {
            "id": str(user.id),
            "name": user.name,
            "discriminator": user.discriminator,
            "bot": user.bot,
            "created_at": user.created_at.isoformat()
        }
        return [TextContent(
            type="text",
            text=f"User information:\n" +
                 f"Name: {user_info['name']}#{user_info['discriminator']}\n" +
                 f"ID: {user_info['id']}\n" +
                 f"Bot: {user_info['bot']}\n" +
                 f"Created: {user_info['created_at']}"
        )]
    
    elif name == "relationship_analysis":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        limit = min(int(arguments.get("limit", 10)), 100)
        fetch_users = arguments.get("fetch_users", False)

        messages = []
        async for message in channel.history(limit=limit):
            reaction_data = []
            for reaction in message.reactions:
                emoji_str = str(reaction.emoji.name) if hasattr(reaction.emoji, 'name') and reaction.emoji.name \
                    else str(reaction.emoji.id) if hasattr(reaction.emoji, 'id') else str(reaction.emoji)
                users = []
                if fetch_users:
                    try:
                        reacted_users = []
                        async for user_obj in reaction.users():
                            reacted_users.append(f"{user_obj.name}#{user_obj.discriminator}")
                        users = reacted_users
                    except Exception as e:
                        logger.warning(f"Failed to fetch reaction users for emoji {emoji_str}: {e}")
                reaction_data.append({
                    "emoji": emoji_str,
                    "count": reaction.count,
                    "users": users
                })
            messages.append({
                "author": str(message.author),
                "author_id": str(message.author.id),
                "content": message.content,
                "mentions": [str(u.id) for u in message.mentions],
                "timestamp": message.created_at.isoformat(),
                "is_reply": bool(message.reference),
                "reactions": reaction_data
            })
        messages.sort(key=lambda m: m["timestamp"])
        def format_reaction_rel(r):
            if r["users"]:
                return f"{r['emoji']}({r['count']}): {', '.join(r['users'])}"
            else:
                return f"{r['emoji']}({r['count']})"
        summary_text = "Message Summary:\n\n"
        for m in messages:
            summary_text += (
                f"{m['author']} ({m['timestamp']}): {m['content']}\n"
                f"Reactions: {', '.join([format_reaction_rel(r) for r in m['reactions']]) if m['reactions'] else 'No reactions'}\n\n"
            )
        user_ids = {msg["author_id"]: msg["author"] for msg in messages}
        interaction_counts = defaultdict(lambda: {"mentions": 0, "replies": 0, "reactions": 0})
        for msg in messages:
            sender = msg["author"]
            mentions = msg["mentions"]
            is_reply = msg["is_reply"]
            for mentioned_id in mentions:
                mentioned_name = user_ids.get(mentioned_id)
                if mentioned_name and mentioned_name != sender:
                    pair = tuple(sorted([sender, mentioned_name]))
                    interaction_counts[pair]["mentions"] += 1
            if is_reply:
                for other_author_id in user_ids:
                    if user_ids[other_author_id] != sender :
                        if message.reference and message.reference.message_id:
                            pair = tuple(sorted([sender, user_ids[other_author_id]]))
                            interaction_counts[pair]["replies"] += 1
            for other_author_id in user_ids:
                other_author_name = user_ids[other_author_id]
                if other_author_name != sender:
                    pair = tuple(sorted([sender, other_author_name]))
                    interaction_counts[pair]["reactions"] += sum(r["count"] for r in msg["reactions"] if r["count"] > 0)

        def generate_description(pair, stats):
            total = stats["mentions"] + stats["replies"] + stats["reactions"]
            if total == 0:
                return None
            parts = []
            if stats["mentions"]: parts.append(f"{stats['mentions']} mention(s)")
            if stats["replies"]: parts.append(f"{stats['replies']} reply/replies")
            if stats["reactions"]: parts.append(f"{stats['reactions']} reaction point(s)")
            return f"{pair[0]} <-> {pair[1]}: Interaction via {', '.join(parts)}."

        relationship_descriptions = [
            desc for desc in (generate_description(pair, stats)
            for pair, stats in interaction_counts.items()) if desc is not None
        ]
        if not relationship_descriptions:
            relationship_report = "Pairwise Relationship Estimates:\nNo significant pairwise interactions observed."
        else:
            relationship_report = "Pairwise Relationship Estimates:\n" + "\n".join(relationship_descriptions)

        full_output = (
            "Relationship Analysis Report\n"
            "=============================\n\n"
            f"Total messages analyzed: {len(messages)}\n\n"
            f"{summary_text}\n"
            f"{relationship_report}"
        )
        return [TextContent(type="text", text=full_output)]

    elif name == "summarize_text":
        # Summarizes messages and sends them as a DM to a returning user
        user_id_str = arguments["user_id"]
        try:
            user_id = int(user_id_str)
        except ValueError:
            logger.warning(f"Invalid user_id format for summarize_text: {user_id_str}")
            return [TextContent(type="text", text="Invalid user_id format")]

        user = await discord_client.fetch_user(user_id)
        # Find a guild shared by both the bot and the user
        guild = next((g for g in discord_client.guilds if g.get_member(user_id)), None)
        
        if not guild:
            logger.warning(f"User {user_id} not found in any mutual guilds with the bot for summarize_text.")
            return [TextContent(type="text", text="User not in a mutual guild or bot cannot see user.")]

        buckets = await _gather_messages(guild)
        raw_parts = []
        if buckets["notice"]:
            raw_parts.append("【Rules/Notice】\n" + "\n".join(buckets["notice"]))
        if buckets["calendar"]:
            raw_parts.append("【Calendar】\n" + "\n".join(buckets["calendar"]))
        if buckets["talk"]:
            raw_parts.append("【General/Chat】\n" + "\n".join(buckets["talk"]))
        
        if not raw_parts:
            # No new messages, just greet the user
            try:
                await user.send("Welcome back! No new important messages to summarize since you were last active.")
                return [TextContent(type="text", text="No messages to summarize, DM sent.")]
            except discord.Forbidden:
                logger.warning(f"DM send failed to {user_id}: Bot is not allowed to DM this user.")
                return [TextContent(type="text", text=f"DM failed: Bot cannot DM user {user_id}")]
            except Exception as e:
                logger.warning(f"DM send failed to {user_id}: {e}")
                return [TextContent(type="text", text=f"DM failed: {e}")]

        raw = "\n\n".join(raw_parts)
        summary = await _llm_summarize(raw)

        try:
            await user.send(f"Welcome back! Here's a summary of what happened while you were away:\n\n{summary}")
            logger.info(f"Summary DM sent to user {user_id}")
            return [TextContent(type="text", text="Summary DM sent")]
        except discord.Forbidden:
            logger.warning(f"DM send failed to {user_id}: Bot is not allowed to DM this user.")
            return [TextContent(type="text", text=f"DM failed: Bot cannot DM user {user_id}")]
        except Exception as e:
            logger.warning(f"DM send failed to {user_id}: {e}")
            return [TextContent(type="text", text=f"DM failed: {e}")]
    
    # Unhandled tool name: returns an empty list, as required by the MCP protocol.
    return []

# ─────────────────── Discord Events ───────────────────
@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """
    Fired when a member's status is updated (for example, when coming online).
    If the user comes online after 7+ days of inactivity, trigger the summary DM.
    """
    # Detect when a user transitions from offline to online
    if before.status == discord.Status.offline and after.status != discord.Status.offline:
        uid = str(after.id)
        now = datetime.utcnow()
        
        last_seen_dt = None
        if uid in last_seen:
            try:
                last_seen_dt = datetime.fromisoformat(last_seen[uid])
            except ValueError:
                logger.warning(f"Could not parse ISO format date for user {uid}: {last_seen[uid]}")

        # If user was inactive for more than the threshold, send them a summary
        if not last_seen_dt or (now - last_seen_dt).days >= INACTIVE_DAYS:
            logger.info(f"User {after.display_name} ({uid}) came online after {INACTIVE_DAYS}+ days. Triggering summary.")
            try:
                await app.request_tool("summarize_text", {"user_id": uid})
            except Exception as e:
                logger.error(f"Error requesting summarize_text tool for user {uid}: {e}")
        
        last_seen[uid] = now.isoformat()
        _save_last_seen()

async def main():
    """Main entrypoint for the Discord bot + MCP server."""
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

