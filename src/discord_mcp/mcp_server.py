"""
Import all required libraries, both standard and external.
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
# Initialization, Logging, and Discord/MCP Setup
##################################################

# Set up global logging for the whole service.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_mcp_server")

# Get Discord bot token from environment for security reasons.
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# Initialize Discord bot with intent permissions.
# message_content: To read messages; members/presences: To track member activity.
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True
bot = commands.Bot(command_prefix="!", intents=intents)

# MCP server setup for tool-based communication.
app = Server("discord_server")

# Store Discord client object so it can be used in tool handlers.
discord_client: Optional[commands.Bot] = None

##################################################
# File Storage, Constants, and Channel Category Helpers
##################################################

# Constants for saving last-seen times, defining inactivity, message limits.
LAST_SEEN_FILE        = "last_seen.json"     # File path to store last seen times for users.
INACTIVE_DAYS         = 7                    # Days before user is considered "inactive".
SUMMARY_MAX_CHARS     = 1000                 # Max chars for the summary passed to LLM.

# Message limits by category: notice/calendar/talk.
NOTICE_LIMIT, CAL_LIMIT, TALK_LIMIT = 30, 30, 300

# Channel categorization keywords.
NOTICE_CH   = {"rules", "rule", "notice", "announcements"}
CALENDAR_CH = {"calendar", "calender", "schedule"}
TALK_CH     = {"general", "chat", "talk", "discussion"}

# In-memory last_seen: maps user_id (str) -> ISO timestamp.
last_seen: Dict[str, str] = {}

def _load_last_seen():
    """
    Loads last seen data from JSON file into memory.
    Keeps track of user inactivity for the bot, across restarts.
    """
    global last_seen
    if os.path.exists(LAST_SEEN_FILE):
        try:
            with open(LAST_SEEN_FILE, encoding="utf-8") as f:
                last_seen = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {LAST_SEEN_FILE}: {e}")

def _save_last_seen():
    """
    Saves the current last_seen state to disk for persistence.
    Protects against repeated notification of the same user.
    """
    try:
        with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(last_seen, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save {LAST_SEEN_FILE}: {e}")

##################################################
# Channel Categorization and Message Gathering
##################################################

async def _categorize(ch: discord.TextChannel) -> Optional[str]:
    """
    Categorizes a channel by its name using the predefined keyword sets.
    Returns category name (notice/calendar/talk) or None if no match.
    """
    n = ch.name.lower()
    if n in NOTICE_CH:   return "notice"
    if n in CALENDAR_CH: return "calendar"
    if n in TALK_CH:     return "talk"
    return None

async def _gather_messages(guild: discord.Guild) -> Dict[str, List[str]]:
    """
    For the provided guild:
    - Collects up to N most recent messages from each categorized channel.
    - Returns a dictionary with keys "notice", "calendar", "talk", each mapping to a list of "Author: Content" strings.
    - Reverses messages so that oldest come first, for easier summary context.
    - Skips channels with missing permissions or other errors.
    """
    buckets = {"notice": [], "calendar": [], "talk": []}
    for ch in guild.text_channels:
        kind = await _categorize(ch)
        if not kind:
            continue
        limit = {"notice": NOTICE_LIMIT, "calendar": CAL_LIMIT, "talk": TALK_LIMIT}[kind]
        try:
            channel_messages = []
            async for msg in ch.history(limit=limit):
                channel_messages.append(f"{msg.author.display_name}: {msg.content}")
            # Reverse order to chronological for summarization
            buckets[kind].extend(reversed(channel_messages))
        except Exception as e:
            logger.warning(f"History fetch fail for channel {ch.name}: {e}")
    return buckets

##################################################
# Discord Ready Event and Client Protection
##################################################

@bot.event
async def on_ready():
    """
    Discord event: Fired once the bot is ready and fully connected.
    - Stores the Discord client globally for use in tools.
    - Loads last seen data for member inactivity tracking.
    """
    global discord_client
    discord_client = bot
    logger.info(f"Log as {bot.user.name}")
    _load_last_seen()  # Load state for inactivity logic

def require_discord_client(func):
    """
    Decorator to prevent running commands/tools before the Discord client is initialized.
    Used to make sure all tool calls wait for full bot readiness.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("client not ready")
        return await func(*args, **kwargs)
    return wrapper

##################################################
# MCP Tool Definitions and Main Tool Dispatcher
##################################################

@app.list_tools()
async def list_tools() -> List[Tool]:
    """
    Registers all tools exposed to the MCP server.
    These tools are available to be called by LLMs or other agents.
    - Each Tool: name, description, and input schema (for validation)
    - The summarize_text tool is used for gathering categorized messages for a given user.
    """
    return [
        Tool(
            name="get_server_info",
            description="Get information about a Discord server",
            inputSchema={
                "type": "object",
                "properties": {"server_id": {"type": "string", "description": "Discord server (guild) ID"}},
                "required": ["server_id"]
            }
        ),
        Tool(
            name="role_analysis",
            description="Analyze user role in discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Discord channel ID"},
                    "limit": {"type": "number", "description": "Messages to fetch (max 100)", "minimum": 1, "maximum": 100}
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="get_user_info",
            description="Get information about a Discord user",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string", "description": "Discord user ID"}},
                "required": ["user_id"]
            }
        ),
        Tool(
            name="list_members",
            description="Get a list of members in a server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Discord server (guild) ID"},
                    "limit": {"type": "number", "description": "Maximum number of members to fetch", "minimum": 1, "maximum": 1000}
                },
                "required": ["server_id"]
            }
        ),
        Tool(
            name="list_channels",
            description="Get a list of all text channels in a Discord server (guild), including their names and channel IDs.",
            inputSchema={
                "type": "object",
                "properties": {"server_id": {"type": "string", "description": "Discord server (guild) ID"}},
                "required": ["server_id"]
            }
        ),
        Tool(
            name="relationship_analysis",
            description="Analyze relationships between users in a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Discord channel ID"},
                    "limit": {"type": "number", "description": "Messages to analyze (max 100)", "minimum": 1, "maximum": 100}
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="summarize_text",
            description="Collect recent messages (Rules/Calendar/Chat) and return raw text",
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
    """
    Main entrypoint for all MCP tool calls.
    - Receives tool name and arguments.
    - Dispatches to the relevant handler for each tool.
    - Returns a list of TextContent (MCP standard).
    - All tools except summarize_text have their own specialized handlers.
    - summarize_text: gathers and returns categorized messages for the given user, to be summarized externally by LLM.
    """

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
        fetch_users = arguments.get("fetch_reaction_users", False)

        messages = []
        
        async for message in channel.history(limit=limit):
            reaction_data = []

            for reaction in message.reactions:
                emoji_str = str(reaction.emoji.name) if hasattr(reaction.emoji, 'name') and reaction.emoji.name else str(
                    reaction.emoji.id) if hasattr(reaction.emoji, 'id') else str(reaction.emoji)

                reaction_info = {
                    "emoji": emoji_str,
                    "count": reaction.count
                }

                logger.error(f"Emoji: {emoji_str}")
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
                        users = [f"{user.name}#{user.discriminator}" for user in await reaction.users().flatten()]
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

        def format_reaction(r):
            if r["users"]:
                return f"{r['emoji']}({r['count']}): {', '.join(r['users'])}"
            else:
                return f"{r['emoji']}({r['count']})"

        summary_text = "Message Summary:\n\n"
        for m in messages:
            summary_text += (
                f"{m['author']} ({m['timestamp']}): {m['content']}\n"
                f"Reactions: {', '.join([format_reaction(r) for r in m['reactions']]) if m['reactions'] else 'No reactions'}\n\n"
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
                for other in user_ids.values():
                    if other != sender:
                        pair = tuple(sorted([sender, other]))
                        interaction_counts[pair]["replies"] += 1
            for other in user_ids.values():
                if other != sender:
                    pair = tuple(sorted([sender, other]))
                    interaction_counts[pair]["reactions"] += sum(r["count"] for r in msg["reactions"])

        def generate_description(pair, stats):
            total = stats["mentions"] + stats["replies"] + stats["reactions"]
            if total == 0:
                return f"{pair[0]} <-> {pair[1]}: No significant interaction observed."
            parts = []
            if stats["mentions"]:
                parts.append("mentions")
            if stats["replies"]:
                parts.append("replies")
            if stats["reactions"]:
                parts.append("reactions")
            return f"{pair[0]} <-> {pair[1]}: Interaction via {', '.join(parts)}."

        relationship_descriptions = [
            generate_description(pair, stats)
            for pair, stats in interaction_counts.items()
        ]

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
        user_id_str = arguments["user_id"]
        try:
            user_id = int(user_id_str)
        except ValueError:
            logger.warning(f"Invalid user_id format for summarize_text: {user_id_str}")
            return [TextContent(type="text", text="Invalid user_id format")]

        guild = next((g for g in discord_client.guilds if g.get_member(user_id)), None)
        if not guild:
            logger.warning(f"User {user_id} not found in any mutual guilds with the bot.")
            return [TextContent(type="text", text="User not in a mutual guild or bot cannot see user.")]

        buckets = await _gather_messages(guild)
        raw_parts: List[str] = []
        if buckets["notice"]:
            raw_parts.append("【Rules/Notice】\n"  + "\n".join(buckets["notice"]))
        if buckets["calendar"]:
            raw_parts.append("【Calendar】\n"     + "\n".join(buckets["calendar"]))
        if buckets["talk"]:
            raw_parts.append("【General/Chat】\n" + "\n".join(buckets["talk"]))

        if not raw_parts:
            return [TextContent(type="text", text="No new messages to summarize.")]

        raw_text = "\n\n".join(raw_parts)
        return [TextContent(type="text", text=raw_text)]
        
    elif name == "list_members":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        limit = min(int(arguments.get("limit", 100)), 1000)
    
        members = []
        async for member in guild.fetch_members(limit=limit):
            members.append({
                "id": str(member.id),
                "name": member.name,
                "nick": member.nick,
                "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                "roles": [str(role.id) for role in member.roles[1:]]
            })
    
        return [TextContent(
            type="text",
            text=f"Server Members ({len(members)}):\n" + 
             "\n".join(f"{m['name']} (ID: {m['id']}, Roles: {', '.join(m['roles'])})" for m in members)
        )]
        
    elif name == "list_channels":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))

        all_text_channels = [
            ch for ch in discord_client.get_all_channels()
            if isinstance(ch, discord.TextChannel) and ch.guild.id == guild.id
        ]
        channel_lines = [f"{ch.name} (ID: {ch.id})" for ch in all_text_channels]
        return [TextContent(
            type="text",
            text="Text Channels:\n" + "\n".join(channel_lines)
        )]
    
    return []

##################################################
# Discord Event: Member Update (Status Change)
##################################################

@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """
    Discord event: Fired when a member's status changes (e.g., offline to online).
    - If user comes online after a period of inactivity (INACTIVE_DAYS or more),
      trigger the summarize_text tool to collect recent important messages for that user.
    - Does not send a DM; the summary will be handled elsewhere.
    - Always updates the last_seen timestamp for the user.
    """
    if before.status == discord.Status.offline and after.status != discord.Status.offline:
        uid = str(after.id)
        now = datetime.utcnow()

        last_seen_dt = None
        if uid in last_seen:
            try:
                last_seen_dt = datetime.fromisoformat(last_seen[uid])
            except ValueError:
                logger.warning(f"Could not parse ISO date for {uid}: {last_seen[uid]}")

        if not last_seen_dt or (now - last_seen_dt).days >= INACTIVE_DAYS:
            logger.info(f"User {after.display_name} ({uid}) came online after {INACTIVE_DAYS}+ days. Triggering summary.")
            try:
                await app.request_tool("summarize_text", {"user_id": uid})
            except Exception as e:
                logger.error(f"Error requesting summarize_text for {uid}: {e}")

        last_seen[uid] = now.isoformat()
        _save_last_seen()

##################################################
# Main Application Entrypoint (Runs Discord bot & MCP server)
##################################################

async def main():
    """
    Entrypoint for the application:
    - Starts the Discord bot in the background.
    - Starts the MCP server for handling tool calls via stdio.
    - Keeps both running as long as the program is alive.
    """
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
