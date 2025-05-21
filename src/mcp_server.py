"""
import external library
"""
import asyncio
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps

import discord
from discord.ext import commands
from mcp.server import Server
from mcp.types import Tool, TextContent, EmptyResult
from mcp.server.stdio import stdio_server
##################################################
"""
Initial Constructor
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_mcp_server")

#Discord bot setup
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

app = Server("discord_server")
discord_client = None
##################################################

@bot.event
async def on_ready():
    global discord_client
    discord_client = bot
    logger.info(f"Log as {bot.user.name}")

def require_discord_client(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("client not ready")
        return await func(*args, **kwargs)
    return wrapper

async def list_tools() -> List[Tool]:
    return[
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
                ##"server": {
                  ##  "type": "string",
                    ##"description": 'Server name or ID (optional if bot is only in one server)',
                ##},
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

    )
    ]

@app.call_tool()
@require_discord_client
async def call_tools(name: str, arguments: Any) -> List[TextContent]:

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

        fetch_users = arguments.get("fetch_reaction_users", False)  # Only fetch users if explicitly requested

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

                logger.error(f"Emoji: {emoji_str}")

                reaction_data.append(reaction_info)

            messages.append({
                "id": str(message.id),
                "author": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "reactions": reaction_data  # Add reactions to message dict
            })

        # Helper function to format reactions

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

        messages = []
        
        async for message in channel.history(limit=limit):
            
            messages.append({
                "author": str(message.author),
                "author_id": str(message.author.id),
                "content": message.content,
                "mentions": [str(u.id) for u in message.mentions],
                "timestamp": message.created_at.isoformat(),
                "is_reply": bool(message.reference)
            })

        messages.sort(key=lambda m: m["timestamp"])




async def main():
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initializtion_option()
        )

if __name__ == "__main__":
    asyncio.run(main())