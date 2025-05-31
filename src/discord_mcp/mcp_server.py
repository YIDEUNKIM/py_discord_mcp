import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from functools import wraps

import discord
from discord.ext import commands
from mcp.server import Server # MCP server framework
from mcp.types import Tool, TextContent, EmptyResult # MCP type definitions
from mcp.server.stdio import stdio_server # For running MCP server over stdio

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_mcp_server")

# Load Discord token from environment variable
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# Define Discord bot intents
intents = discord.Intents.default()
intents.message_content = True # Required to read message content
intents.members = True       # Required for member presence and updates
intents.presences = True     # Required for member status updates (e.g., online/offline)
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize MCP server application
app = Server("discord_server")
discord_client: Optional[commands.Bot] = None # Global variable to hold the bot client

# --- Configuration Constants ---
USER_LAST_SEEN_FILE = "user_last_seen_data.json" # File to store user last seen timestamps

# Keywords to identify channels by category for summarization
CATEGORY_KEYWORDS_CONFIG = {
    "notice": ["rules", "rule", "notice", "announcement", "announcements"],
    "schedule": ["schedule", "calendar", "event", "events", "calender"],
    "talk": ["general", "chat", "talk", "discussion", "main", "lounge", "community"],
}
SUMMARY_INACTIVITY_DAYS = 7 # Threshold for sending a summary DM (7 days of inactivity)

# Message and character limits for summary generation
NOTICE_MAX_MSGS = 30
SCHEDULE_MAX_MSGS = 30
TALK_MAX_MSGS = 300
NOTICE_SECTION_INPUT_MAX_CHARS = 7000
SCHEDULE_SECTION_INPUT_MAX_CHARS = 7000
TALK_SECTION_INPUT_MAX_CHARS = 15000
PER_MESSAGE_CHAR_LIMIT_DEFAULT = 250 # Default character limit for a single message in summary input
TARGET_SUMMARY_OUTPUT_LENGTH_CHARS = 1000 # Target length for the LLM-generated summary
MAX_OVERALL_LLM_PROMPT_CHARS = 30000 # Maximum characters for the entire prompt sent to LLM for summarization
DISCORD_DM_MAX_LENGTH = 2000 # Discord's character limit for DMs

MESSAGE_COUNT_THRESHOLD_FOR_ISSUE_LABELING = 50 # Number of messages to accumulate before triggering issue labeling

# --- Utility Functions for JSON I/O ---
def _load_json_sync(path: str, default: Any) -> Any:
    """Synchronously loads JSON from a file. Returns default if file not found or error."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError): # Catch potential errors during file read or JSON parsing
            pass
    return default

def _save_json_sync(path: str, obj: Any):
    """Synchronously saves an object to a JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except OSError: # Catch potential errors during file write
        pass

def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Converts a datetime object to an ISO 8601 string format."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else None

def _iso_to_dt(iso: Optional[str]) -> Optional[datetime]:
    """Converts an ISO 8601 string to a datetime object."""
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S") if iso else None
    except ValueError: # Handle cases where the string is not a valid ISO date format
        return None

async def _load_json_async(path: str, default: Any) -> Any:
    """Asynchronously loads JSON from a file using executor for non-blocking I/O."""
    return await bot.loop.run_in_executor(None, _load_json_sync, path, default)

async def _save_json_async(path: str, obj: Any):
    """Asynchronously saves an object to a JSON file using executor."""
    await bot.loop.run_in_executor(None, _save_json_sync, path, obj)

# In-memory caches
last_seen_user_map: Dict[str, datetime] = {} # Cache for user last seen times {user_id: datetime}
channel_msg_counter: Dict[int, int] = {}     # Cache for message counts per channel {channel_id: count}
channel_last_msg_id: Dict[int, int] = {}     # Cache for last message ID per channel {channel_id: message_id}

# --- Discord Bot Event Handlers ---
@bot.event
async def on_ready():
    """Event handler for when the bot successfully connects to Discord."""
    global discord_client
    discord_client = bot # Make the bot client available globally
    await _initialize_last_seen_cache() # Load user last seen data from file
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")

def require_discord_client(func):
    """Decorator to ensure the Discord client is ready before executing a function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client or not bot.is_ready():
            raise RuntimeError("Discord client not ready or not connected.")
        return await func(*args, **kwargs)
    return wrapper

# --- MCP Tool Definitions ---
@app.list_tools()
async def list_tools() -> List[Tool]:
    """Returns a list of tools available for the MCP server."""
    return [
        Tool(
            name="get_server_info",
            description="Get information about a Discord server",
            inputSchema={
                "type": "object",
                "properties": {"server_id": {"type": "string", "description": "Discord server (guild) ID"}},
                "required": ["server_id"],
            },
        ),
        Tool(
            name="role_analysis",
            description="Analyze user role in a Discord channel based on message history",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Discord channel ID"},
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to fetch (max 100)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["channel_id"],
            },
        ),
        Tool(
            name="get_user_info",
            description="Get information about a Discord user",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string", "description": "Discord user ID"}},
                "required": ["user_id"],
            },
        ),
        Tool(
            name="relationship_analysis",
            description="Analyze relationships between users in a Discord channel (placeholder)",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Discord channel ID"},
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to analyze (max 100)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["channel_id"],
            },
        ),
        Tool(
            name="summarize_text",
            description="Summarize provided text via LLM (delegated to MCP server)",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_to_summarize": {"type": "string"},
                    "target_char_length": {"type": "number"},
                },
                "required": ["text_to_summarize", "target_char_length"],
            },
        ),
        Tool(
            name="issue_message_labeling",
            description="Label issue messages from a list via LLM (delegated to MCP server)",
            inputSchema={
                "type": "object",
                "properties": {"messages_for_labeling": {"type": "string"}}, # A string of concatenated messages
                "required": ["messages_for_labeling"],
            },
        ),
    ]

# --- MCP Tool Call Handler ---
@app.call_tool()
@require_discord_client # Ensure bot is ready before handling tool calls
async def call_tools(name: str, arguments: Any) -> List[TextContent]: # type: ignore
    """Handles tool calls from the MCP server."""
    if name == "get_server_info":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        info = {
            "Name": guild.name,
            "ID": guild.id,
            "OwnerID": guild.owner_id,
            "MemberCount": guild.member_count,
            "CreatedAt": guild.created_at.isoformat(),
            "PremiumTier": guild.premium_tier,
            "ExplicitContentFilter": str(guild.explicit_content_filter),
            "Description": guild.description or "N/A",
        }
        text = "Server Information:\n" + "\n".join(f"- {k}: {v}" for k, v in info.items())
        return [TextContent(type="text", text=text)]

    elif name == "role_analysis":
        channel_id_str = arguments["channel_id"]
        channel = await discord_client.fetch_channel(int(channel_id_str))
        if not isinstance(channel, discord.TextChannel):
            return [TextContent(type="text", text=f"Error: Channel ID {channel_id_str} is not a text channel.")]
        
        limit = min(int(arguments.get("limit", 10)), 100) # Default to 10, max 100
        messages_data = []
        async for m in channel.history(limit=limit):
            reactions = [{"emoji": str(r.emoji), "count": r.count} for r in m.reactions]
            messages_data.append(
                {
                    "id": str(m.id),
                    "author": str(m.author),
                    "content": m.content,
                    "timestamp": m.created_at.isoformat(),
                    "reactions": reactions,
                }
            )
        
        def fmt_reaction(r): return f"{r['emoji']}({r['count']})"
        text_report = "\n".join(
            f"{msg['author']} ({msg['timestamp']}): {msg['content']}\n"
            f"Reactions: {', '.join(fmt_reaction(r) for r in msg['reactions']) if msg['reactions'] else 'None'}"
            for msg in messages_data
        )
        return [TextContent(type="text", text=f"Retrieved {len(messages_data)} messages for role analysis:\n\n{text_report}")]

    elif name == "get_user_info":
        user = await discord_client.fetch_user(int(arguments["user_id"]))
        info = {
            "ID": user.id,
            "UsernameTag": str(user), # username#discriminator
            "Bot": user.bot,
            "CreatedAt": user.created_at.isoformat(),
        }
        text = "User Information:\n" + "\n".join(f"- {k}: {v}" for k, v in info.items())
        return [TextContent(type="text", text=text)]

    elif name == "relationship_analysis":
        # This is a placeholder implementation as per the original code.
        channel_id_str = arguments["channel_id"]
        channel = await discord_client.fetch_channel(int(channel_id_str))
        if not isinstance(channel, discord.TextChannel):
            return [TextContent(type="text", text=f"Error: Channel ID {channel_id_str} is not a text channel.")]
        # Placeholder response
        return [TextContent(
            type="text",
            text="Relationship Analysis (Placeholder):\n"
                 "-'A' and 'B' have a mutually trusting relationship. \n"
                 "-'A' and 'C' often conflict but respect each other.\n"
                 "-'B' and 'C' are comfortable and close with each other."
        )]

    elif name == "summarize_text":
        # This tool delegates the actual summarization to the LLM via MCP.
        # This bot only acknowledges the request.
        text_to_summarize = arguments.get("text_to_summarize", "")
        target_char_len = int(arguments.get("target_char_length", TARGET_SUMMARY_OUTPUT_LENGTH_CHARS))
        if not text_to_summarize.strip():
            return [TextContent(type="text", text="[summarize_text] No content provided to summarize.")]
        # The actual summarization happens on the MCP server side.
        return [TextContent(type="text", text=f"[summarize_text] Summarization request received for {len(text_to_summarize)} chars -> target {target_char_len} chars. Awaiting LLM processing.")]

    elif name == "issue_message_labeling":
        # This tool delegates actual labeling to the LLM via MCP.
        # This bot only acknowledges the request.
        messages_for_labeling = arguments.get("messages_for_labeling", "")
        if not messages_for_labeling.strip():
            return [TextContent(type="text", text="[issue_message_labeling] No messages provided for labeling.")]
        # The actual labeling happens on the MCP server side.
        return [TextContent(type="text", text="[issue_message_labeling] Labeling request received. Awaiting LLM processing.")]

    return [EmptyResult()] # Return empty result if tool name is not recognized

# --- Core Logic for Summaries and Issue Labeling ---

def _find_channels_by_category(guild: discord.Guild, category_key: str) -> List[discord.TextChannel]:
    """Finds text channels in a guild that match keywords for a given category."""
    keywords = CATEGORY_KEYWORDS_CONFIG.get(category_key, [])
    if not keywords: return []
    
    matching_channels = []
    for channel in guild.text_channels:
        # Check if bot has permission to read messages in the channel
        if channel.permissions_for(guild.me).read_messages:
            if any(keyword in channel.name.lower() for keyword in keywords):
                matching_channels.append(channel)
    return matching_channels

async def _format_messages(
    channels: List[discord.TextChannel],
    since: datetime, # Timestamp to fetch messages after
    max_messages_to_fetch: int, # Max messages to fetch across all provided channels for this category
    per_message_char_limit: int, # Max characters for a single message's content
    total_chars_limit_for_section: int, # Max total characters for this section's input to LLM
) -> str:
    """
    Formats messages from specified channels for LLM input.
    Collects messages, truncates long ones, and respects overall character/message limits.
    """
    collected_lines: List[str] = []
    current_char_count = 0
    current_message_count = 0

    for channel in channels:
        if current_message_count >= max_messages_to_fetch or current_char_count >= total_chars_limit_for_section:
            break
        
        # Ensure bot has permission to read message history
        if not channel.permissions_for(channel.guild.me).read_message_history:
            logger.warning(f"Missing read_message_history permission for channel: {channel.name} ({channel.id})")
            continue

        try:
            # Fetch messages: limit is per channel, oldest_first=False means newest first
            async for message in channel.history(limit=max_messages_to_fetch, after=since, oldest_first=False):
                if current_message_count >= max_messages_to_fetch or current_char_count >= total_chars_limit_for_section:
                    break
                
                if message.author.bot or not message.content.strip(): # Skip bot messages or empty messages
                    continue

                author_name = message.author.display_name
                content = message.content.strip().replace("\n", " ") # Normalize newlines

                # Truncate individual message content if it exceeds per_message_char_limit
                if len(content) > per_message_char_limit:
                    content = content[:per_message_char_limit - 3] + "..."
                
                formatted_line = f"{author_name}: {content}"
                
                # Calculate projected character count if this line is added
                # Add 1 for newline character if list is not empty
                projected_total_chars = current_char_count + len(formatted_line) + (1 if collected_lines else 0)

                if projected_total_chars > total_chars_limit_for_section:
                    # If adding the full line exceeds limit, try to fit a truncated version
                    remaining_space = total_chars_limit_for_section - current_char_count - (1 if collected_lines else 0)
                    if remaining_space > 20: # Only add if there's meaningful space left
                        collected_lines.append(formatted_line[:remaining_space - 3] + "...")
                    current_char_count = total_chars_limit_for_section # Mark as full
                    break # Stop collecting from this channel and subsequent ones
                
                collected_lines.append(formatted_line)
                current_char_count = projected_total_chars
                current_message_count += 1
        
        except discord.Forbidden:
            logger.warning(f"Forbidden to read history from channel: {channel.name} ({channel.id})")
        except Exception as e:
            logger.error(f"Error fetching history from {channel.name}: {e}", exc_info=True)
            
    # Messages are fetched newest first, so reverse to get chronological order for the summary
    return "\n".join(reversed(collected_lines))

async def _initialize_last_seen_cache():
    """Loads the last seen user data from the JSON file into memory."""
    global last_seen_user_map
    raw_data = await _load_json_async(USER_LAST_SEEN_FILE, {})
    last_seen_user_map = {
        user_id: dt
        for user_id, timestamp_str in raw_data.items()
        if (dt := _iso_to_dt(timestamp_str)) is not None # Load and convert valid timestamps
    }
    logger.info(f"Initialized last_seen_user_map with {len(last_seen_user_map)} entries.")

async def _persist_last_seen_cache():
    """Saves the current in-memory last seen user data to the JSON file."""
    await _save_json_async(USER_LAST_SEEN_FILE, {
        uid: _dt_to_iso(dt) for uid, dt in last_seen_user_map.items()
    })

async def _send_summary_dm(member: discord.Member, guild: discord.Guild, last_seen_datetime: Optional[datetime]):
    """
    Generates and sends a summary DM to a member who has been inactive.
    """
    now_utc = datetime.utcnow()
    # Define the cutoff date for messages to include in the summary
    summary_cutoff_date = now_utc - timedelta(days=SUMMARY_INACTIVITY_DAYS)

    # Fetch messages for "notice" category
    notice_channels = _find_channels_by_category(guild, "notice")
    notice_text_block = await _format_messages(
        notice_channels,
        summary_cutoff_date,
        NOTICE_MAX_MSGS,
        PER_MESSAGE_CHAR_LIMIT_DEFAULT,
        NOTICE_SECTION_INPUT_MAX_CHARS,
    )

    # Fetch messages for "schedule" category
    schedule_channels = _find_channels_by_category(guild, "schedule")
    schedule_text_block = await _format_messages(
        schedule_channels,
        summary_cutoff_date,
        SCHEDULE_MAX_MSGS,
        PER_MESSAGE_CHAR_LIMIT_DEFAULT,
        SCHEDULE_SECTION_INPUT_MAX_CHARS,
    )

    # Fetch messages for "talk" category
    talk_channels = _find_channels_by_category(guild, "talk")
    talk_text_block = await _format_messages(
        talk_channels,
        summary_cutoff_date,
        TALK_MAX_MSGS,
        PER_MESSAGE_CHAR_LIMIT_DEFAULT - 50, # Slightly reduce per-message limit for general talk to allow more messages
        TALK_SECTION_INPUT_MAX_CHARS,
    )

    # Combine all text blocks for LLM input
    summary_blocks = []
    if notice_text_block:
        summary_blocks.append(f"[Recent Notices]\n{notice_text_block}")
    if schedule_text_block:
        summary_blocks.append(f"[Schedules / Events]\n{schedule_text_block}")
    if talk_text_block:
        summary_blocks.append(f"[General Discussions]\n{talk_text_block}")
    
    llm_prompt_input = "\n\n".join(summary_blocks)
    if not llm_prompt_input.strip(): # If no messages were found in any category
        llm_prompt_input = f"No significant server activity found in the last {SUMMARY_INACTIVITY_DAYS} days."

    # Truncate overall LLM prompt if it exceeds the absolute maximum
    if len(llm_prompt_input) > MAX_OVERALL_LLM_PROMPT_CHARS:
        llm_prompt_input = llm_prompt_input[:MAX_OVERALL_LLM_PROMPT_CHARS - 3] + "..."
        logger.warning(f"Truncated LLM prompt for user {member.id} due to exceeding MAX_OVERALL_LLM_PROMPT_CHARS.")

    # Call the MCP tool to get the summary from LLM
    logger.info(f"Requesting summary for user {member.id} (text length: {len(llm_prompt_input)})")
    tool_response_parts = await call_tools(
        "summarize_text",
        {"text_to_summarize": llm_prompt_input, "target_char_length": TARGET_SUMMARY_OUTPUT_LENGTH_CHARS},
    )
    
    generated_summary = "Summary currently unavailable. Please check the channels directly." # Default if LLM fails
    if tool_response_parts and isinstance(tool_response_parts[0], TextContent) and tool_response_parts[0].text:
        # This currently gets the acknowledgment message, not the actual summary from LLM.
        # The actual summary would come from the MCP server's LLM processing.
        # For now, we use the placeholder text from the tool stub.
        # In a real scenario, this `tool_response_parts[0].text` would be the LLM's summarized output.
        generated_summary = tool_response_parts[0].text 


    days_absent_str = "several"
    if last_seen_datetime:
        days_absent = (now_utc - last_seen_datetime).days
        days_absent_str = str(days_absent) if days_absent > 0 else "less than a"

    # Prepare and send the DM
    dm_intro = (
        f"Welcome back to **{guild.name}**! "
        f"(It seems you were away for ~{days_absent_str} days)\n\n"
        f"Here's a brief summary of what happened while you were away:\n"
    )
    dm_final_content = (dm_intro + generated_summary)[:DISCORD_DM_MAX_LENGTH]

    try:
        await member.send(dm_final_content)
        logger.info(f"Sent welcome back summary DM to {member.name} ({member.id})")
    except discord.Forbidden:
        logger.warning(f"Cannot send DM to {member.name} ({member.id}). DMs might be disabled.")
    except Exception as e:
        logger.error(f"Failed to send summary DM to {member.name} ({member.id}): {e}", exc_info=True)

# --- More Discord Bot Event Handlers ---

@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """
    Event handler for when a member's presence or profile changes.
    Used to detect when a user comes online after a period of inactivity.
    """
    if after.id == bot.user.id or after.bot: # Ignore bot's own updates or other bots
        return

    # Check if user came online from an offline status
    if before.status != discord.Status.online and after.status == discord.Status.online:
        user_id_str = str(after.id)
        current_time_utc = datetime.utcnow()
        
        last_seen_time = last_seen_user_map.get(user_id_str)
        
        # Send summary if user was inactive for SUMMARY_INACTIVITY_DAYS or never seen before
        if last_seen_time is None or (current_time_utc - last_seen_time) >= timedelta(days=SUMMARY_INACTIVITY_DAYS):
            logger.info(f"User {after.name} ({after.id}) came online after {SUMMARY_INACTIVITY_DAYS}+ days of inactivity (or first time seen).")
            await _send_summary_dm(after, after.guild, last_seen_time)
        
        # Update last seen time for the user
        last_seen_user_map[user_id_str] = current_time_utc
        await _persist_last_seen_cache() # Save updated cache to file

@bot.event
async def on_message(message: discord.Message):
    """
    Event handler for every new message.
    Used for issue labeling in "talk" channels.
    """
    await bot.process_commands(message) # Allow bot commands to be processed

    if message.author.bot or not message.guild: # Ignore DMs and messages from bots
        return

    # Check if the message is in a "talk" category channel
    is_talk_channel = any(keyword in message.channel.name.lower() for keyword in CATEGORY_KEYWORDS_CONFIG.get("talk", []))
    if not is_talk_channel:
        return

    channel_id = message.channel.id
    channel_last_msg_id.setdefault(channel_id, 0)
    channel_msg_counter.setdefault(channel_id, 0)

    # Increment message counter for new messages only (handles potential restarts/reconnections)
    if message.id > channel_last_msg_id[channel_id]:
        channel_msg_counter[channel_id] += 1
        channel_last_msg_id[channel_id] = message.id

    # If message threshold is reached, trigger issue labeling
    if channel_msg_counter[channel_id] >= MESSAGE_COUNT_THRESHOLD_FOR_ISSUE_LABELING:
        channel_msg_counter[channel_id] = 0 # Reset counter for this channel
        
        recent_messages_for_labeling: List[discord.Message] = []
        try:
            # Fetch the last N messages (oldest_first=True to get them in chronological order for LLM)
            async for msg_history_item in message.channel.history(
                limit=MESSAGE_COUNT_THRESHOLD_FOR_ISSUE_LABELING, oldest_first=True
            ):
                if not msg_history_item.author.bot: # Exclude bot messages from labeling input
                    recent_messages_for_labeling.append(msg_history_item)
        except discord.Forbidden:
            logger.warning(f"Forbidden to read history for issue labeling in channel: {message.channel.name} ({channel_id})")
            return
        except Exception as e:
            logger.error(f"Error fetching history for issue labeling in {message.channel.name}: {e}", exc_info=True)
            return
        
        # Ensure we only use the most recent N messages if history fetched more (e.g. due to bot messages being skipped)
        recent_messages_for_labeling = recent_messages_for_labeling[-MESSAGE_COUNT_THRESHOLD_FOR_ISSUE_LABELING:]

        # Prepare prompt for LLM: enumerated list of "Author: Content"
        llm_labeling_prompt = "\n".join(
            f"{idx + 1}. {msg.author.display_name}: {msg.content.strip().replace(chr(10), ' ')}" # Replace newlines with spaces
            for idx, msg in enumerate(recent_messages_for_labeling)
        )

        if not llm_labeling_prompt.strip(): # If no valid messages found
            return

        logger.info(f"Requesting issue labeling for channel {message.channel.name} ({channel_id}) for {len(recent_messages_for_labeling)} messages.")
        # Call MCP tool for issue labeling
        tool_labeling_response_parts = await call_tools(
            "issue_message_labeling", 
            {"messages_for_labeling": llm_labeling_prompt}
        )

        if not tool_labeling_response_parts or not isinstance(tool_labeling_response_parts[0], TextContent) or not tool_labeling_response_parts[0].text:
            logger.warning("No valid response from issue_message_labeling tool.")
            return

        # The actual labeling and emoji suggestion would come from the MCP server's LLM.
        # This bot expects a JSON string like: '[{"index": 1, "suggested_emoji": "🎉"}, ...]'
        # Current placeholder response from call_tools is just an acknowledgement.
        # For a real implementation, the MCP server would return the JSON.
        # For demonstration, let's assume the placeholder response can be parsed or modified to test this.
        try:
            # Example: If the stub returned '{"labels": [{"index": 1, "suggested_emoji": "👍"}]}'
            # For now, we'll just log the text that would be parsed
            logger.info(f"Received from issue_message_labeling tool: {tool_labeling_response_parts[0].text}")
            
            # Attempt to parse the response as JSON (this will likely fail with the current placeholder)
            suggested_labels = json.loads(tool_labeling_response_parts[0].text)
            
            if isinstance(suggested_labels, list):
                for label_info in suggested_labels:
                    msg_index = label_info.get("index") # 1-based index
                    emoji_to_add = label_info.get("suggested_emoji")

                    if isinstance(msg_index, int) and emoji_to_add and 0 < msg_index <= len(recent_messages_for_labeling):
                        target_message_to_react = recent_messages_for_labeling[msg_index - 1]
                        try:
                            await target_message_to_react.add_reaction(emoji_to_add)
                            logger.info(f"Added reaction {emoji_to_add} to message {target_message_to_react.id}")
                        except discord.HTTPException as e: # Reaction might be invalid, message deleted, etc.
                            logger.warning(f"Failed to add reaction {emoji_to_add} to message {target_message_to_react.id}: {e}")
                        except Exception as e:
                             logger.error(f"Unexpected error adding reaction: {e}", exc_info=True)
        except json.JSONDecodeError:
            # This is expected if the `issue_message_labeling` tool stub doesn't return valid JSON
            logger.info("Could not decode JSON from issue_message_labeling tool response (this is expected if using placeholder).")
        except Exception as e:
            logger.error(f"Error processing labels for issue messages: {e}", exc_info=True)


# --- Main Execution ---
async def main():
    """Main function to start the Discord bot and the MCP server."""
    # Start the Discord bot in a separate task
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    
    # Start the MCP server using stdio
    # It will listen for tool calls from another process (e.g., an LLM agent controller)
    async with stdio_server() as (reader, writer):
        initialization_options = app.create_initialization_options()
        await app.run(reader, writer, initialization_options)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Discord MCP server.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
