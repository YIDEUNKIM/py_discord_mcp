"""From Discord data to Claude Desktop for MCP"""
from . import mcp_server
import asyncio
import warnings
import tracemalloc

__version__ = "0.1.0" # MCP library version

def main():
    """Main entry point"""

    #tracemalloc: Tracks memory allocations and helps find where memory blocks are allocated
    tracemalloc.start() # start trace memory

    # Suppress PyNaCl warning since we don't use voice features
    warnings.filterwarnings('ignore', module='discord.client', message='PyNaCl not installed')
    # PyNaCl is an encryption library required for Discord voice features

    try:
        # run async
        asyncio.run(mcp_server.main())
    except KeyboardInterrupt:
        print("Shut Down by KeyboardInterrupt...")

    except Exception as e:
        print(f"Error running as {e}")
        raise

# Show important items at package level
__all__ = ['main', 'mcp_server']


