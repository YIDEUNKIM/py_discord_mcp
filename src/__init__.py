"""From Discord data to Claude Desktop for MCP"""
from . import mcp_server
import asyncio
import warnings
import tracemalloc

__version__ = "0.1.0" # MCP library version

def main():
    """Main entry point"""

    """tracemalloc: 메모리 할당 추적하고 메모리 블록이 할당된 위치를 찾는 데 도움을 줌"""
    tracemalloc.start() # 메모리 추적 시작

    # 코드 실행
    # ...

    """PyNacl이 설치되지 않았다는 경고 메시지가 표시되는 것을 방지함"""
    warnings.filterwarnings('ignore', module='discord.client', message='PyNaCl not installed')
    # PyNacl은 Discord 음성 기능에 필요한 암호화 라이브러리

    try:
        # 비동기 실행 제어
        asyncio.run(McpServer.main())
    except KeyboardInterrupt:
        print("Shut Down by KeyboardInterrupt...")

    except Exception as e:
        print(f"Error running as {e}")
        raise

# package level 에서 중요한 items 를 미리 보여줌
__all__ = ['main', 'mcp_server.py']


