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


##################################################


async def ready():
    pass

def require_discord_client():
    pass

async def list_tools() -> List[Tool]:
    pass

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