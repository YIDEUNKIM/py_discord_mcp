import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from functools import wraps

import discord
from discord.ext import commands
import anthropic

from mcp.server import Server
from mcp.types import Tool, TextContent, EmptyResult
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_mcp_server")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
claude_client: Optional[anthropic.Anthropic] = None
if CLAUDE_API_KEY:
    try:
        claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        logger.info("Anthropic Claude client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic Claude client: {e}. Summarization feature will be impacted.")
else:
    logger.warning("CLAUDE_API_KEY is not set. LLM summarization tool ('summarize_text') will not be available.")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

app = Server("discord_server")
discord_client = None

LAST_SEEN_FILE = "last_seen.json"
MEMBER_LOG_FILE = "member_log.json"

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def dt_to_str(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else None

def str_to_dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S") if s else None

async def load_json_async(path, default):
    return await bot.loop.run_in_executor(None, load_json, path, default)

async def save_json_async(path, obj):
    await bot.loop.run_in_executor(None, save_json, path, obj)

last_seen: Dict[str, datetime] = {}
member_log: List[Dict[str, Any]] = []

async def initialize_data():
    global last_seen, member_log
    raw_last_seen = await load_json_async(LAST_SEEN_FILE, {})
    last_seen = {uid: str_to_dt(ts) for uid, ts in raw_last_seen.items() if str_to_dt(ts) is not None}
    member_log = await load_json_async(MEMBER_LOG_FILE, [])

async def save_last_seen_async_wrapper():
    data_to_save = {uid: dt_to_str(dt) for uid, dt in last_seen.items()}
    await save_json_async(LAST_SEEN_FILE, data_to_save)

async def save_member_log_async_wrapper():
    await save_json_async(MEMBER_LOG_FILE, member_log)

def log_member_event(guild_id, user_id, username, event):
    member_log.append({
        "guild_id": str(guild_id),
        "user_id": str(user_id),
        "username": username,
        "event": event,
        "time": dt_to_str(datetime.utcnow())
    })

def get_recent_member_changes(guild_id, since: datetime):
    logs = [
        log for log in member_log
        if log["guild_id"] == str(guild_id) and str_to_dt(log["time"]) >= since
    ]
    out = []
    for log in logs:
        t = str_to_dt(log["time"]).strftime("%Y-%m-%d %H:%M")
        out.append(f"{t}: {log['username']} {log['event']}")
    return "\n".join(out) if out else "No recent join/leave activity."

def channel_name_candidates():
    return {
        "notice": ["notice", "announcement", "announcements", "rule", "rules"],
        "schedule": ["schedule", "calendar", "event", "events"],
        "talk": ["general", "chat", "talk", "discussion", "main", "lounge", "community"]
    }

def find_channel_by_names(guild, purpose):
    candidates = channel_name_candidates()[purpose]
    for name in candidates:
        for ch in guild.text_channels:
            if name in ch.name.lower():
                return ch
    return None

def get_role_send_permissions(channel):
    allowed_roles = []
    denied_roles = []
    for role in channel.guild.roles:
        perm = channel.permissions_for(role)
        if perm.send_messages:
            allowed_roles.append(role)
        else:
            denied_roles.append(role)
    return allowed_roles, denied_roles

def is_channel_posting_restricted(channel: discord.TextChannel) -> bool:
    return not channel.permissions_for(channel.guild.default_role).send_messages

async def find_special_purpose_channel(guild: discord.Guild, purpose: str, restricted_by_perms: bool) -> Optional[discord.TextChannel]:
    channel = find_channel_by_names(guild, purpose)
    if channel:
        if restricted_by_perms:
            if is_channel_posting_restricted(channel):
                return channel
        else:
            if not is_channel_posting_restricted(channel):
                return channel
    candidate_channels: List[discord.TextChannel] = []
    for ch in guild.text_channels:
        if not ch.permissions_for(guild.me).read_messages:
            continue
        if restricted_by_perms == is_channel_posting_restricted(ch):
            candidate_channels.append(ch)
    if candidate_channels:
        return candidate_channels[0]
    return None

async def find_talk_channel_by_activity(guild: discord.Guild, since: datetime) -> Optional[discord.TextChannel]:
    potential_talk_channels = [
        ch for ch in guild.text_channels
        if ch.permissions_for(guild.me).read_messages and
           ch.permissions_for(guild.me).read_message_history and
           not is_channel_posting_restricted(ch)
    ]
    potential_talk_channels.sort(key=lambda c: c.last_message_id or 0, reverse=True)
    for channel_to_check in potential_talk_channels[:5]:
        try:
            async for msg in channel_to_check.history(limit=50, after=since, oldest_first=False):
                return channel_to_check
        except Exception:
            continue
    if potential_talk_channels:
        return potential_talk_channels[0]
    return None

async def collect_channel_messages(
    channel: Optional[discord.TextChannel], 
    since: datetime, 
    max_messages_to_collect: int = 20, 
    per_message_char_limit: int = 200,
    overall_section_char_limit: int = 3000
    ) -> str:
    collected_texts = []
    current_total_chars = 0
    if channel is None or not channel.permissions_for(channel.guild.me).read_message_history:
        return ""
    try:
        num_collected = 0
        async for msg in channel.history(limit=max_messages_to_collect * 2, after=since, oldest_first=False):
            if num_collected >= max_messages_to_collect:
                break
            if msg.content.strip() and not msg.author.bot:
                author_name = msg.author.display_name
                content_text = msg.content.strip().replace("\n", " ")
                if len(content_text) > per_message_char_limit:
                    content_text = content_text[:per_message_char_limit-3] + "..."
                formatted_message = f"{author_name}: {content_text}"
                if current_total_chars + len(formatted_message) + (1 if collected_texts else 0) > overall_section_char_limit:
                    if overall_section_char_limit - current_total_chars > 20:
                        can_add_len = overall_section_char_limit - current_total_chars - (1 if collected_texts else 0) - 3
                        collected_texts.append(formatted_message[:can_add_len] + "...")
                    break 
                collected_texts.append(formatted_message)
                current_total_chars += len(formatted_message) + (1 if len(collected_texts)>1 else 0)
                num_collected += 1
    except Exception:
        pass
    return "\n".join(reversed(collected_texts))

async def collect_summary_input(guild: discord.Guild, since: Optional[datetime]) -> str:
    cutoff = since or (datetime.utcnow() - timedelta(days=31))
    NOTICE_SECTION_MAX_CHARS = 4000
    SCHEDULE_SECTION_MAX_CHARS = 4000
    TALK_SECTION_MAX_CHARS = 6000
    notice_ch = await find_special_purpose_channel(guild, "notice", restricted_by_perms=True)
    schedule_ch = await find_special_purpose_channel(guild, "schedule", restricted_by_perms=True)
    talk_ch_by_name = find_channel_by_names(guild, "talk")
    if talk_ch_by_name and not is_channel_posting_restricted(talk_ch_by_name):
        talk_ch = talk_ch_by_name
    else:
        talk_ch = await find_talk_channel_by_activity(guild, cutoff)
    notice_msgs = await collect_channel_messages(notice_ch, cutoff, max_messages_to_collect=15, per_message_char_limit=300, overall_section_char_limit=NOTICE_SECTION_MAX_CHARS)
    schedule_msgs = await collect_channel_messages(schedule_ch, cutoff, max_messages_to_collect=15, per_message_char_limit=300, overall_section_char_limit=SCHEDULE_SECTION_MAX_CHARS)
    talk_msgs = await collect_channel_messages(talk_ch, cutoff, max_messages_to_collect=40, per_message_char_limit=200, overall_section_char_limit=TALK_SECTION_MAX_CHARS)
    member_changes_str = get_recent_member_changes(str(guild.id), cutoff)
    prompt_sections = []
    if notice_msgs: prompt_sections.append(f"[Server Notices - Recent Updates]\n{notice_msgs}")
    if schedule_msgs: prompt_sections.append(f"[Server Schedules - Upcoming Events]\n{schedule_msgs}")
    if talk_msgs: prompt_sections.append(f"[General Discussions - Hot Topics]\n{talk_msgs}")
    if member_changes_str and member_changes_str != "No recent join/leave activity.":
        prompt_sections.append(f"[Member Activity - Recent Changes]\n{member_changes_str}")
    if not prompt_sections:
        return "No significant server activity found in the last month."
    final_llm_prompt = "\n\n".join(prompt_sections)
    if len(final_llm_prompt) > 25000:
        final_llm_prompt = final_llm_prompt[:25000]
    return final_llm_prompt

def require_discord_client(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("client not ready")
        return await func(*args, **kwargs)
    return wrapper

async def request_llm_summary_via_mcp(full_llm_prompt: str, desired_output_length: int = 900) -> str:
    tool_name = "summarize_text"
    arguments = {
        "text_to_summarize": full_llm_prompt, 
        "target_char_length": desired_output_length
    }
    results = await call_tools(tool_name, arguments)
    if results and isinstance(results, list) and results[0] and hasattr(results[0], "text"):
        llm_raw_response_text = results[0].text.strip()
        if len(llm_raw_response_text) > desired_output_length:
            return llm_raw_response_text[:desired_output_length-3] + "..."
        return llm_raw_response_text
    return "[Summary unavailable]"

async def generate_summary_with_llm(guild: discord.Guild, since: Optional[datetime]) -> str:
    prompt = await collect_summary_input(guild, since)
    if prompt == "No significant server activity found in the last month.":
        return prompt
    summary = await request_llm_summary_via_mcp(prompt, desired_output_length=900)
    return summary

@bot.event
async def on_ready():
    global discord_client
    discord_client = bot
    logger.info(f"Log as {bot.user.name}")
    await initialize_data()

@bot.event
async def on_member_join(member):
    log_member_event(member.guild.id, member.id, str(member), "joined")
    await save_member_log_async_wrapper()

@bot.event
async def on_member_remove(member):
    log_member_event(member.guild.id, member.id, str(member), "left")
    await save_member_log_async_wrapper()

@bot.event
async def on_member_update(before, after):
    if after.id == bot.user.id:
        return
    if before.status != discord.Status.online and after.status == discord.Status.online:
        user_id = str(after.id)
        now = datetime.utcnow()
        last_seen_time = last_seen.get(user_id)
        if after.bot:
            last_seen[user_id] = now
            await save_last_seen_async_wrapper()
            return
        if last_seen_time is None or (now - last_seen_time) >= timedelta(days=31):
            days_inactive_str = (now - last_seen_time).days if last_seen_time else "first_time_tracked"
            summary = await generate_summary_with_llm(after.guild, last_seen_time)
            dm_intro = f"Welcome back to {after.guild.name}, {after.mention}! (You've been away for approx. {days_inactive_str} days)\nHere is a summary of recent server activity:\n\n"
            final_dm = dm_intro + summary
            if len(final_dm) > 2000:
                summary = summary[:2000 - len(dm_intro) - 10] + "..."
                final_dm = dm_intro + summary
            try:
                await after.send(final_dm)
            except discord.Forbidden:
                pass
            except Exception:
                pass
        last_seen[user_id] = now
        await save_last_seen_async_wrapper()

# MCP tool definitions and handlers (unchanged)
async def list_tools() -> List[Tool]:
    return [
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
            name="summarize_text",
            description="Summarize a given text to be less than a target character length using Claude LLM.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_to_summarize": {
                        "type": "string",
                        "description": "The combined text from server activity to be summarized."
                    },
                    "target_char_length": {
                        "type": "number",
                        "description": "The target maximum character length for the final summary from LLM."
                    }
                },
                "required": ["text_to_summarize", "target_char_length"]
            }
        )
    ]

@app.call_tool()
@require_discord_client
async def call_tools(name: str, arguments: Any) -> List[TextContent]:
    if name == "summarize_text":
        if not claude_client:
            return [TextContent(type="text", text="[LLM summarizer unavailable]")]
        input_prompt = arguments.get("text_to_summarize", "")
        llm_output_char_target = int(arguments.get("target_char_length", 900))
        if not input_prompt.strip():
            return [TextContent(type="text", text="No content to summarize.")]
        claude_model = "claude-3-haiku-20240307"
        estimated_max_output_tokens = min(4000, int(llm_output_char_target * 0.8) + 100)
        system_prompt = (
            "You are a helpful assistant specializing in summarizing Discord server activity. "
            "Provide a concise, friendly, and informative overview for users returning after an extended absence. "
            "Focus on key discussions, important announcements, and significant events. "
            "Do not include section headers. Strictly follow the character limit."
        )
        user_content = (
            f"Please summarize the following Discord server activity. "
            f"The summary must be less than {llm_output_char_target} characters.\n\n"
            f"--- Server Activity Text to Summarize ---\n"
            f"{input_prompt}\n"
            f"--- End of Server Activity Text ---"
        )
        try:
            response = await bot.loop.run_in_executor(
                None, 
                claude_client.messages.create,
                {
                    "model": claude_model,
                    "max_tokens": estimated_max_output_tokens,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}]
                }
            )
            llm_generated_text = ""
            if response.content and isinstance(response.content, list) and \
               len(response.content) > 0 and response.content[0].type == "text":
                llm_generated_text = response.content[0].text.strip()
            if len(llm_generated_text) > llm_output_char_target:
                llm_generated_text = llm_generated_text[:llm_output_char_target-3] + "..."
            return [TextContent(type="text", text=llm_generated_text)]
        except Exception:
            return [TextContent(type="text", text="[LLM Summarizer Error]")]
    elif name == "get_server_info":
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
        messages = []
        async for message in channel.history(limit=limit):
            reaction_data = []
            for reaction in message.reactions:
                emoji_str = str(reaction.emoji.name) if hasattr(reaction.emoji, 'name') and reaction.emoji.name \
                    else str(reaction.emoji.id) if hasattr(reaction.emoji, 'id') else str(reaction.emoji)
                reaction_data.append({
                    "emoji": emoji_str,
                    "count": reaction.count
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
        return [TextContent( 
            type="text",
            text="Relationship Analysis:\n"
                 "-'A'와 'B'는 서로 의지하며 신뢰하는 관계입니다. \n"
                 "-'A'와 'C'는 종종 대립하나 서로 존중하는 관계입니다.\n"
                 "-'B'와 'C'는 서로 편하게 대하며 친밀한 관계입니다."
        )]

channel_message_counter: Dict[int, int] = {}
channel_last_message_id: Dict[int, int] = {}

@bot.event
async def on_message(message):
    await bot.process_commands(message)
    if message.author.bot:
        return
    channel_id = message.channel.id
    if channel_id not in channel_message_counter:
        channel_message_counter[channel_id] = 0
        channel_last_message_id[channel_id] = 0
    if message.id > channel_last_message_id[channel_id]:
        channel_message_counter[channel_id] += 1
        channel_last_message_id[channel_id] = message.id
    if channel_message_counter[channel_id] >= 50:
        channel_message_counter[channel_id] = 0
        messages = []
        async for msg in message.channel.history(limit=50, oldest_first=True):
            if not msg.author.bot:
                messages.append(msg)
        if not messages:
            return
        prompt_messages = "\n".join(
            [f"{i+1}. {msg.author.display_name}: {msg.content}" for i, msg in enumerate(messages)]
        )
        system_prompt = (
            "You are an expert at analyzing Discord chat for community impact. "
            "Given the list of messages, label only those messages that clearly stand out as [funny, informative, touching, argument]. "
            "If none stand out, return an empty JSON array []. "
            "For each labeled message, return JSON item: {index, category, suggested_emoji}. "
            "Emojis: 😂 for funny, ✅ for informative, 🥹 for touching, 🔥 for argument. "
            "If no message is clearly suitable, return []. Only JSON, no explanation."
        )
        user_prompt = (
            "Analyze the following 50 Discord messages and classify each if suitable:\n"
            + prompt_messages
            + "\n\nReturn in JSON format as explained above."
        )
        if not claude_client:
            return
        try:
            response = await bot.loop.run_in_executor(
                None,
                claude_client.messages.create,
                {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 2000,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
            )
            parsed = None
            for c in response.content:
                try:
                    parsed = json.loads(c.text.strip())
                    break
                except Exception:
                    continue
            if not parsed or not isinstance(parsed, list) or not parsed:
                return
        except Exception:
            return
        for item in parsed:
            idx = item.get("index")
            emoji = item.get("suggested_emoji")
            if not (idx and emoji):
                continue
            try:
                msg = messages[int(idx)-1]
                await msg.add_reaction(emoji)
            except discord.Forbidden:
                continue
            except Exception:
                continue

async def main():
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
