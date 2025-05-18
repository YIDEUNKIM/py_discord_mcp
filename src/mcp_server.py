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
bot = commands.Bot(commands_prefix="!", intents=intents)

app = Server("discord_server")
discord_client = None
##################################################

@bot.event
async def ready():
    global dicord_client
    discord_client = bot
    logger.info(f"Log as {bot.user.name}")

def require_discord_client():
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("client not ready")
        return await func(*args, **kwargs)
    return wrapper

async def list_tools() -> List[Tool]:
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
        name="read_messages",
        description="Read recent messages from a channel",
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
async def call_tools(name: str, arg: Any) -> List[TextContent]:
    pass



def main():
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initializtion_option()
        )

if __name__ == "__main__":
    asyncio.run(main())